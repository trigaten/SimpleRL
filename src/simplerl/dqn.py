from copy import deepcopy
from simplerl.core import Agent, Policy
from simplerl.buffers import BasicExperienceBuffer
from simplerl.policies import DiscreteRandomPolicy
from simplerl.utils import discount_rewards
import random

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
        # discounted_rewards = discount_rewards()
        pass


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