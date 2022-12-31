import random

import torch

from simplerl.core import Policy


class DiscreteRandomPolicy(Policy):
    """Random policy that returns a discrete random action.

    :param num_actions: The integer number of actions to choose from.
    """
    def __init__(self, num_actions:int):
        self.num_actions = num_actions

    def __call__(self, obs):
        return random.randint(0, self.num_actions - 1)
        
class MaxQPolicy(Policy):
    def __init__(self, net, num_actions:int):
        self.net = net
        self.num_actions = num_actions

    def calc_q_values(self, obs:torch.FloatTensor):
        return self.net(obs)

    def __call__(self, obs):
        with torch.no_grad():
            q_values = self.calc_q_values(obs)
            return q_values.argmax(1).item()