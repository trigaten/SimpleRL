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
from torch.utils.tensorboard import SummaryWriter

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
        start_learning_step:int, update_freq:int, target_update_freq:int,
        log_path=None):

        self.experience_buffer = buffer
        self.gamma = gamma
        self.epsilon = epsilon
        self.policy = policy
        # TODO: make this perform a load, not deepcopy...
        self.target_policy = deepcopy(policy)
        self.num_actions = policy.num_actions
        self.random_policy = DiscreteRandomPolicy(num_actions=self.num_actions)
        self.loss = nn.MSELoss()
        self.optimizer = optimizer
        self.start_learning_step = start_learning_step
        self.update_freq = update_freq
        self.target_update_freq = target_update_freq
        self.episode = 0

        if log_path:
            self.logger = SummaryWriter(log_path)

    def __call__(self, obs):
        # TODO: refactor
        # to device
        obs = torch.FloatTensor(obs)
        obs = torch.unsqueeze(obs, -1)
        obs = torch.transpose(obs, 0, 1)
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
        
        if self.logger:
            self.logger.add_scalar('loss', loss.detach().item(), self.episode)
            
        return loss.item()
    
    def calc_loss(self, sample_experience):
        """Compute the differentiable Q learning loss from a batch of experiences.
        
        :param sample_experience: a batch of experiences
        
        """
        # unpack the batch of experiences
        states, actions, rewards, next_states, dones = unzip_experience_buffer(sample_experience)
        
        # compute the target q values for the loss equation (non-differentiable)
        # TODO: remove other no grad usage
        with torch.no_grad():
            target_values = self.compute_target_values(rewards, next_states, dones)

        # compute the newly estimated q values (differentiable)
        q_values = self.compute_q_values(states, actions)

        # compute the loss
        loss = self.loss(q_values, target_values)

        return loss

    def compute_target_values(self, rewards, next_states, dones):
        next_state_q_values = self.next_state_maxq_values(next_states, dones)
        disc_next_state_q_values = self.gamma * next_state_q_values
        rewards = torch.FloatTensor(rewards)
        target_values = rewards + disc_next_state_q_values
        return target_values

    def next_state_maxq_values(self, next_states:List[np.array], dones:List[bool]):
        """Compute the q values of the next states using the target policy."""
        with torch.no_grad():
            s_ = torch.from_numpy(np.stack(next_states)).float()
            q_values = self.target_policy.calc_q_values(s_)
            selected_q_values = q_values.max(dim=1)
            torch_dones = torch.LongTensor(dones)
            return selected_q_values.values * (1 - torch_dones)

    # TODO: rename?
    def compute_q_values(self, states:List[np.array], actions:List[bool]):
        """Compute the q values of the state action pairs"""
        s = torch.from_numpy(np.stack(states)).float()
        q_values = self.policy.calc_q_values(s)

        actions = torch.LongTensor(actions)

        return q_values.gather(1, actions.unsqueeze(-1)).squeeze()

    def post_episode_stage(self, episodes):
        self.episode+=1
        if episodes >= self.start_learning_step:
            if episodes % self.update_freq == 0:
                self.update()
            if episodes % self.target_update_freq == 0:
                self.target_policy.net.load_state_dict(self.policy.net.state_dict())

    def post_experience(self, state, action, reward, next_state, done):
        self.experience_buffer.add(state, action, reward, next_state, done)