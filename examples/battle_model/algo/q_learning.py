import os
import torch
import numpy as np

from . import base
from . import tools


class MFQ(base.ValueNet):
    def __init__(self, handle, env, sub_len, eps=1.0, update_every=5, memory_size=2**10, batch_size=64):
        super().__init__(env, handle)

        config = {
            'max_len': memory_size,
            'batch_size': batch_size,
            'obs_shape': self.view_space,
            'feat_shape': self.feature_space,
            'act_n': self.num_actions,
            'use_mean': True,
            'sub_len': sub_len
        }

        self.train_ct = 0
        self.replay_buffer = tools.MemoryGroup(**config)

    def flush_buffer(self, **kwargs):
        self.replay_buffer.push(**kwargs)

    def train(self):
        self.replay_buffer.tight()
        batch_num = self.replay_buffer.get_batch_num()

        for i in range(batch_num):
            obs, feat, acts, act_prob, obs_next, feat_next, act_prob_next, rewards, dones, masks = self.replay_buffer.sample()
            target_q = self.calc_target_q(obs=obs_next, feature=feat_next, rewards=rewards, dones=dones, prob=act_prob_next)
            loss, q = super().train(state=[obs, feat], target_q=target_q, prob=act_prob, acts=acts, masks=masks)

            self.update()

            # if i % 50 == 0:
            #     print('[*] LOSS:', loss, '/ Q:', q)

    def save(self, dir_path, step=0):
        file_path = os.path.join(dir_path, "mfq_{}".format(step))
        torch.save(self.Q.state_dict(), file_path)
        print("[*] Model saved at: {}".format(file_path))

    def load(self, dir_path, step=0):
        file_path = os.path.join(dir_path, "mfq_{}".format(step))
        self.Q.load_state_dict(torch.load(file_path))
        print("[*] Loaded model from {}".format(file_path))

