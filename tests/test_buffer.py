from inspect import isabstract

from numpy import size
import gym
import torch.nn as nn

from simplerl.core import Agent, Policy
from simplerl.buffers import BasicExperienceBuffer
from simplerl.policies import DiscreteRandomPolicy
from simplerl.core import Experience

from simplerl.dqn import DQN

class TestBuffer:
    def test_add(self):
        exp_buffer = BasicExperienceBuffer(size=10, batch_size=2)
        exp_buffer.add(1, 1, 1, 1)

        # check the length is correct
        assert len(exp_buffer) == 1

        for _ in range(9):
            exp_buffer.add(0, 0, 0, 0)

        # check the length is correct after adding 9 more experiences
        assert len(exp_buffer) == 10
        
        # check that the first experience added is still there
        assert exp_buffer[0].s == 1 and exp_buffer[0].a == 1 and exp_buffer[0].r == 1 and exp_buffer[0].s_ == 1

        exp_buffer.add(0, 0, 0, 0)
        # check that the first experience has been pushed out
        assert exp_buffer[0].s == 0 and exp_buffer[0].a == 0 and exp_buffer[0].r == 0 and exp_buffer[0].s_ == 0

    def test_sample(self):
        exp_buffer = BasicExperienceBuffer(size=10, batch_size=2)
        for i in range(10):
            exp_buffer.add(i, i, i, i)

        # check that the sample is correct
        batch = exp_buffer.sample()
        assert len(batch) == 2
        assert batch[0].s != batch[1].s
                
