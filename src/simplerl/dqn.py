from copy import deepcopy
from typing import List
import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from simplerl.core import Agent, Policy
from simplerl.buffers import BasicExperienceBuffer
from simplerl.policies import DiscreteRandomPolicy
from simplerl.utils import discount_rewards, unzip_experience_buffer



class DQN(Agent):
    """
    :param buffer: the experience buffer to add to/sample from
    :param gamma: the discount factor
    :param epsilon: the epsilon value for the epsilon-greedy policy
    :param policy: the policy to use for the agent
    :param start_learning: the number of steps before the agent starts learning
    :param update_freq: the number of steps between each update step
    :param target_update_freq: the number of steps between each target network update
    """
    def __init__(self, 
        buffer:BasicExperienceBuffer, 
        gamma:float, epsilon:float, 
        policy:Policy, optimizer:optim.Optimizer, 
        start_learning_step:int, update_freq:int, target_update_freq:int):

        self.experience_buffer = buffer
        self.gamma = gamma
        self.epsilon = epsilon
        self.policy = policy
        self.target_policy = deepcopy(policy)
        self.num_actions = policy.num_actions
        self.random_policy = DiscreteRandomPolicy(num_actions=self.num_actions)
        self.loss = nn.MSELoss()
        self.optimizer = optimizer
        self.start_learning_step = start_learning_step
        self.update_freq = update_freq
        self.target_update_freq = target_update_freq

    def __call__(self, obs):
        if random.random() < self.epsilon:
            return self.random_policy(obs)
        
        return self.policy(obs)

    def update(self):
        # sample a batch of experiences
        sample_experience = self.experience_buffer.sample()
        
        # compute loss
        loss = self.calc_loss(sample_experience)

        # backpropogate gradients
        loss.backward()

        # perform an optimization step
        self.optimizer.step()

        # clear gradients
        self.optimizer.zero_grad()
        
        return loss.item()
    
    def calc_loss(self, sample_experience):
        states, actions, rewards, next_states, dones = unzip_experience_buffer(sample_experience)
        # TODO: remove other no grad usage
        with torch.no_grad():
            target_values = self.compute_target_values(rewards, next_states, dones)

        q_values = self.compute_q_values(states, actions)

        loss = self.loss(q_values, target_values)

        return loss

    def compute_target_values(self, rewards, next_states, dones):
        next_state_q_values = self.next_state_maxq_values(next_states, dones)
        disc_next_state_q_values = self.gamma * next_state_q_values
        rewards = torch.FloatTensor(rewards)
        target_values = rewards + disc_next_state_q_values
        return target_values

    def next_state_maxq_values(self, next_states:List[np.array], dones:List[bool]):
        """Compute the q values of the next states"""
        with torch.no_grad():
            s_ = torch.from_numpy(np.stack(next_states)).float()
            q_values = self.target_policy.calc_q_values(s_)
            selected_q_values = q_values.max(dim=1)
            torch_dones = torch.LongTensor(dones)
            return selected_q_values.values * (1 - torch_dones)

    # TODO: rename?
    def compute_q_values(self, states:List[np.array], actions:List[bool]):
        s = torch.from_numpy(np.stack(states)).float()
        q_values = self.target_policy.calc_q_values(s)

        actions = torch.LongTensor(actions)

        return q_values.gather(1, actions.unsqueeze(-1)).squeeze()

    # def step_hook(self):
        

        


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