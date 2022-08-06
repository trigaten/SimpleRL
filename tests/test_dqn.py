import gym
import torch
import torch.nn as nn
import numpy as np

from simplerl.core import Experience
from simplerl.policies import DiscreteRandomPolicy
from simplerl.buffers import BasicExperienceBuffer
from simplerl.dqn import DQN

from simplerl import unzip_experience_buffer, MaxQPolicy

class LinearNet(nn.Module):
    """Dummy network"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 2, bias=False)

    def forward(self, x):
        return self.fc(x)

class TestPolicies:
    def test_random_policy(self):
        policy = DiscreteRandomPolicy(num_actions=9)
        assert policy(0) in range(9)

        total = sum([policy(0) for _ in range(10000)])
        mean = total / 10000
        # mean should be approximately 4
        assert mean > 3.5 and mean < 5.5

class TestDQN:
    def test_basic(self):
        """test basic properties of DQN"""
        # instantiate a DQN agent
        dqn = DQN(buffer=BasicExperienceBuffer(size=10, batch_size=2), gamma=0.9, epsilon=0.1, policy=DiscreteRandomPolicy(num_actions=2))
        # instantiate a gym environment
        env = gym.make('CartPole-v1', new_step_api=True)
        obs = env.reset() 

        # run an episode
        terminated =  False
        truncated =  False
        while not terminated or truncated:
            # get the action from the agent
            action = dqn(obs)
            # step the environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            # add the experience to the buffer
            dqn.experience_buffer.add(obs, action, reward, next_obs)
            # update the new obs
            obs = next_obs

        assert len(dqn.experience_buffer) > 2
        

        class Policy7:
            def __init__(self):
                self.num_actions = 7
            def __call__(self, obs):
                return 7

        dqn = DQN(buffer=BasicExperienceBuffer(size=10, batch_size=2), gamma=0.9, epsilon=0.1, policy=Policy7())
        total_7s = 0
        # test epsilon randomness is working
        for i in range(1000):
            action = dqn(-1)
            if action == 7:
                total_7s+=1

        # should be a bit above 900 7s
        assert total_7s > 860 and total_7s < 960

        buffer = [Experience(0, 1, 1, 0), Experience(0, 0, 0, 0), Experience(0, 1, 1, 0)]
    
    def test_next_state_maxq_values(self):
        net = LinearNet()
        weights  = net.state_dict()
        weights['fc.weight'] = torch.FloatTensor([[2.0],[-1.0]])
        net.load_state_dict(weights)
        policy = MaxQPolicy(net, num_actions=2)
        # instantiate a DQN agent
        buffer = BasicExperienceBuffer(size=10, batch_size=2)
        dqn = DQN(buffer=buffer, gamma=0.9, epsilon=0.1, policy=policy)
        
        buffer.add(np.array([1.0]), 2, 3, np.array([4.0]))
        buffer.add(np.array([1.0]), 2, 3, np.array([-2.0]))
        buffer.add(np.array([1.0]), 2, 3, np.array([1.0]))

        states, actions, rewards, next_states = unzip_experience_buffer(buffer)
        assert torch.equal(dqn.next_state_maxq_values(next_states), torch.Tensor([8.0, 2.0, 2.0]))

        