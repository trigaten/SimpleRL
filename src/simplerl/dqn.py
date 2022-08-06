from copy import deepcopy
from typing import List
import random

import torch
import numpy as np

from simplerl.core import Agent, Policy
from simplerl.buffers import BasicExperienceBuffer
from simplerl.policies import DiscreteRandomPolicy
from simplerl.utils import discount_rewards, unzip_experience_buffer



class DQN(Agent):
    def __init__(self, buffer:BasicExperienceBuffer, gamma:float, epsilon:float, policy:Policy):
        """"""
        self.experience_buffer = buffer
        self.gamma = gamma
        self.epsilon = epsilon
        self.policy = policy
        self.target_policy = deepcopy(policy)
        self.num_actions = policy.num_actions
        self.random_policy = DiscreteRandomPolicy(num_actions=self.num_actions)

    def __call__(self, obs):
        if random.random() < self.epsilon:
            return self.random_policy(obs)
        
        return self.policy(obs)

    def update(self):
        # sample a batch of experiences
        sample_experience = self.experience_buffer.sample()
        states, actions, rewards, next_states = unzip_experience_buffer(sample_experience)

        pass

    def next_state_maxq_values(self, next_states:List[np.array]):
        with torch.no_grad():
            s_ = torch.from_numpy(np.stack(next_states)).float()
            q_values = self.target_policy.calc_q_values(s_)
            max_q_values = q_values.max(dim=1)
            return max_q_values.values


# def dqn_train(env, agent):
#     # "global" state of training
#     training_done = False
#     # how many episodes have been completed thus far
#     episodes_complete = 0
#     while not training_done:
#         obs = env.reset()
#         done = False
#         while not done:
#             obs, reward, done, info = agent(obs)

#         episodes_complete+= 1
#         if episodes_complete > 1000:
#             training_done = True