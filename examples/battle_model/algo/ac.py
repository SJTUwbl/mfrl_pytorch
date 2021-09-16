import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import tools

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Base(nn.Module):
    """docstring for Base"""
    def __init__(self, view_space, feature_space, num_actions, hidden_size):
        super(Base, self).__init__()

        self.view_space = view_space  # view_width * view_height * n_channel
        self.feature_space = feature_space # feature_size
        self.num_actions = num_actions

        # for input_view
        self.l1 = nn.Linear(np.prod(view_space), hidden_size)
        # for input_feature
        self.l2 = nn.Linear(feature_space[0], hidden_size)
        # for input_act_prob
        self.l3 = nn.Linear(num_actions, 64)
        self.l4 = nn.Linear(64, 32)

    def forward(self, input_view, input_feature, input_act_prob):
        # flatten_view = torch.FloatTensor(input_view)
        flatten_view = input_view.reshape(-1, np.prod(self.view_space))
        h_view = F.relu(self.l1(flatten_view))

        h_emb  = F.relu(self.l2(input_feature))

        emb_prob = F.relu(self.l3(input_act_prob))
        dense_prob = F.relu(self.l4(emb_prob))

        concat_layer = torch.cat([h_view, h_emb, dense_prob], dim=1)
        return concat_layer

class Actor(nn.Module):
    """docstring for Actor"""
    def __init__(self, hidden_size, num_actions):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(32 + 2 * hidden_size, hidden_size * 2)
        self.l2 = nn.Linear(hidden_size * 2, num_actions)

    def forward(self, concat_layer):
        dense = F.relu(self.l1(concat_layer))
        policy = F.softmax(self.l2(dense / 0.1), dim=-1)
        policy = policy.clamp(1e-10, 1-1e-10)
        return policy

class Critic(nn.Module):
    """docstring for Critic"""
    def __init__(self, hidden_size):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(32 + 2 * hidden_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, 1)

    def forward(self, concat_layer):
        dense = F.relu(self.l1(concat_layer))
        value = self.l2(dense)
        value = value.reshape(-1)
        return value

class MFAC:
    """docstring for MFAC"""
    def __init__(self, handle, env, value_coef=0.1, ent_coef=0.08, gamma=0.95, batch_size=64, learning_rate=1e-4):
        self.env = env

        self.view_space = env.get_view_space(handle)
        self.feature_space = env.get_feature_space(handle)
        self.num_actions = env.get_action_space(handle)[0]
        self.reward_decay = gamma

        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.value_coef = value_coef  # coefficient of value in the total loss
        self.ent_coef = ent_coef  # coefficient of entropy in the total loss

        # init training buffers
        self.replay_buffer = tools.EpisodesBuffer(use_mean=True)

        hidden_size = 256
        self.base = Base(self.view_space, self.feature_space, self.num_actions, hidden_size).to(device)
        self.actor = Actor(hidden_size, self.num_actions).to(device)
        self.critic = Critic(hidden_size).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.base.parameters(),   'lr': learning_rate},
            {'params': self.actor.parameters(),  'lr': learning_rate},
            {'params': self.critic.parameters(), 'lr': learning_rate}
            ])

    @property
    def vars(self):
        return [self.base, self.actor, self.critic]

    def act(self, **kwargs):
        input_view = torch.FloatTensor(kwargs['state'][0]).to(device)
        input_feature = torch.FloatTensor(kwargs['state'][1]).to(device)
        input_act_prob = torch.FloatTensor(kwargs['prob']).to(device)
        concat_layer = self.base(input_view, input_feature, input_act_prob)
        policy = self.actor(concat_layer)
        action = torch.multinomial(policy, 1)
        action = action.cpu().numpy()
        return action.astype(np.int32).reshape((-1,))

    def train(self):
        # calc buffer size
        n = 0
        # batch_data = sample_buffer.episodes()
        batch_data = self.replay_buffer.episodes()
        self.replay_buffer = tools.EpisodesBuffer(use_mean=True)

        for episode in batch_data:
            n += len(episode.rewards)

        view = torch.FloatTensor(n, *self.view_space).to(device)
        feature = torch.FloatTensor(n, *self.feature_space).to(device)
        action = torch.LongTensor(n).to(device)
        reward = torch.FloatTensor(n).to(device)
        act_prob_buff = torch.FloatTensor(n, self.num_actions).to(device)

        ct = 0
        gamma = self.reward_decay
        # collect episodes from multiple separate buffers to a continuous buffer
        for k, episode in enumerate(batch_data):
            v, f, a, r, prob = episode.views, episode.features, episode.actions, episode.rewards, episode.probs
            v = torch.FloatTensor(v).to(device)
            f = torch.FloatTensor(f).to(device)
            r = torch.FloatTensor(r).to(device)
            a = torch.LongTensor(a).to(device)
            prob = torch.FloatTensor(prob).to(device)

            m = len(episode.rewards)
            assert len(episode.probs) > 0

            concat_layer = self.base(v[-1].reshape(1, -1), f[-1].reshape(1, -1), prob[-1].reshape(1, -1))
            keep = self.critic(concat_layer)[0]

            for i in reversed(range(m)):
                keep = keep * gamma + r[i]
                r[i] = keep

            view[ct:ct + m] = v
            feature[ct:ct + m] = f
            action[ct:ct + m] = a
            reward[ct:ct + m] = r
            act_prob_buff[ct:ct + m] = prob
            ct += m

        assert n == ct

        # train
        concat_layer = self.base(view, feature, act_prob_buff)
        value = self.critic(concat_layer)
        policy = self.actor(concat_layer)

        action_mask = F.one_hot(action, self.num_actions)
        advantage = (reward - value).detach()

        log_policy = torch.log(policy + 1e-6)
        log_prob = torch.sum(log_policy * action_mask, dim=1)

        pg_loss = -torch.mean(advantage * log_prob)
        vf_loss = self.value_coef * torch.mean(torch.square(reward.detach() - value))
        neg_entropy = self.ent_coef * torch.mean(torch.sum(policy * log_policy, dim=1))
        total_loss = pg_loss + vf_loss + neg_entropy

        # train op (clip gradient)
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.base.parameters(), 0.5)
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.optimizer.step()

        print('[*] PG_LOSS:', np.round(pg_loss.item(), 6), '/ VF_LOSS:', np.round(vf_loss.item(), 6), '/ ENT_LOSS:', np.round(neg_entropy.item(), 6), '/ VALUE:', np.mean(value.cpu().detach().numpy()))

    def flush_buffer(self, **kwargs):
        self.replay_buffer.push(**kwargs)

    def save(self, dir_path, step=0):

        model_vars = {
            'base':   self.base.state_dict(),
            'actor':  self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }

        file_path = os.path.join(dir_path, "mfac_{}.pth".format(step))
        torch.save(model_vars, file_path)

        print("[*] Model saved at: {}".format(file_path))

    def load(self, dir_path, step=0):
        file_path = os.path.join(dir_path, "mfac_{}.pth".format(step))
        model_vars = torch.load(file_path)

        self.base.load_state_dict(model_vars['base'])
        self.actor.load_state_dict(model_vars['actor'])
        self.critic.load_state_dict(model_vars['critic'])

        print("[*] Loaded model from {}".format(file_path))
