import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Network(nn.Module):
    """docstring for Network"""
    def __init__(self, view_space, feature_space, num_actions):
        super(Network, self).__init__()

        self.view_space = view_space  # view_width * view_height * n_channel
        self.feature_space = feature_space # feature_size
        self.num_actions = num_actions # n

        self.conv1 = nn.Conv2d(in_channels=view_space[-1], out_channels=16, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        self.l1 = nn.Linear(4*4*32, 256)

        self.l2 = nn.Linear(feature_space[0], 32)

        self.l3 = nn.Linear(num_actions, 64)
        self.l4 = nn.Linear(64, 32)

        self.l5 = nn.Linear(32+32+256, 128)
        self.l6 = nn.Linear(128, 64)
        self.l7 = nn.Linear(64, num_actions)

    def forward(self, obs_input, feat_input, act_prob_input):
        obs_input = obs_input.permute(0, 3, 2, 1)
        x = F.relu(self.conv1(obs_input))
        x = F.relu(self.conv2(x))
        flatten_obs = x.view(x.size(0), -1)
        h_obs = F.relu(self.l1(flatten_obs))

        h_emb = F.relu(self.l2(feat_input))

        prob_emb = F.relu(self.l3(act_prob_input))
        h_act_prob = F.relu(self.l4(prob_emb))

        concat_layer = torch.cat([h_obs, h_emb, h_act_prob], dim=1)
        dense2 = F.relu(self.l5(concat_layer))
        out = F.relu(self.l6(dense2))
        q = self.l7(out)

        return q

class ValueNet:
    def __init__(self, env, handle, learning_rate=1e-4, tau=0.005, gamma=0.95):
        self.env = env

        self.handle = handle
        self.view_space = env.get_view_space(handle)
        assert len(self.view_space) == 3
        self.feature_space = env.get_feature_space(handle)
        self.num_actions = env.get_action_space(handle)[0]

        self.temperature = 0.1

        self.tau = tau
        self.gamma = gamma

        self.Q = Network(self.view_space, self.feature_space, self.num_actions).to(device)
        self.target_Q = Network(self.view_space, self.feature_space, self.num_actions).to(device)

        self.optimizer = optim.Adam(self.Q.parameters(), lr=learning_rate)

    @property
    def vars(self):
        return [self.Q, self.target_Q]

    def calc_target_q(self, **kwargs):
        """Calculate the target Q-value
        kwargs: {'obs', 'feature', 'prob', 'dones', 'rewards'}
        """
        obs_input = torch.FloatTensor(kwargs['obs']).to(device)
        feat_input = torch.FloatTensor(kwargs['feature']).to(device)
        assert kwargs.get('prob', None) is not None
        act_prob_input = torch.FloatTensor(kwargs['prob']).to(device)
        rewards = torch.FloatTensor(kwargs['rewards']).to(device)
        dones = torch.FloatTensor(kwargs['dones']).to(device)

        t_q = self.target_Q(obs_input, feat_input, act_prob_input)
        e_q = self.Q(obs_input, feat_input, act_prob_input)
        act_idx = torch.argmax(e_q, dim=1)
        q_values = t_q[np.arange(len(t_q)), act_idx]

        target_q_value = rewards + (1. - dones) * q_values.reshape(-1) * self.gamma

        return target_q_value.detach().cpu().numpy()

    def update(self):
        """Q-learning update"""
        for param, target_param in zip(self.Q.parameters(), self.target_Q.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def act(self, **kwargs):
        """Act
        kwargs: {'obs', 'feature', 'prob', 'eps'}
        """
        obs_input = torch.FloatTensor(kwargs['state'][0]).to(device)
        feat_input = torch.FloatTensor(kwargs['state'][1]).to(device)
        assert kwargs.get('prob', None) is not None
        assert len(kwargs['prob']) == len(kwargs['state'][0])
        act_prob_input = torch.FloatTensor(kwargs['prob']).to(device)

        temperature = kwargs['eps']
        e_q = self.Q(obs_input, feat_input, act_prob_input)
        actions = F.softmax(e_q/temperature, dim=-1)
        actions = actions.detach().cpu().numpy()
        actions = np.argmax(actions, axis=1).astype(np.int32)
        return actions

    def train(self, **kwargs):
        """Train the model
        kwargs: {'state': [obs, feature], 'target_q', 'prob', 'acts'}
        """
        obs_input = torch.FloatTensor(kwargs['state'][0]).to(device)
        feat_input = torch.FloatTensor(kwargs['state'][1]).to(device)
        target_q_input = torch.FloatTensor(kwargs['target_q']).to(device)
        mask = torch.FloatTensor(kwargs['masks']).to(device)
        assert kwargs.get('prob', None) is not None
        act_prob_input = torch.FloatTensor(kwargs['prob']).to(device)
        act_input = torch.LongTensor(kwargs['acts']).to(device)

        act_one_hot = F.one_hot(act_input, self.num_actions).float()
        e_q = self.Q(obs_input, feat_input, act_prob_input)
        e_q_max = torch.sum(act_one_hot * e_q, dim=1)
        loss = torch.sum(torch.square(target_q_input - e_q_max) * mask) / torch.sum(mask)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), {'Eval-Q': np.round(np.mean(e_q.detach().cpu().numpy()), 6), 'Target-Q': np.round(np.mean(kwargs['target_q']), 6)}
