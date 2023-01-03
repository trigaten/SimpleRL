import gym
import torch
import torch.nn as nn
import numpy as np
from pytest import approx

from simplerl.core import Experience
from simplerl.policies import DiscreteRandomPolicy
from simplerl.buffers import BasicExperienceBuffer
from simplerl.dqn import DQN

from simplerl.core import train

from simplerl import unzip_experience_buffer, MaxQPolicy

class LinearNet(nn.Module):
    """Dummy network"""
    def __init__(self, inputs=1, outputs = 2):
        super().__init__()
        self.fc = nn.Linear(inputs, outputs, bias=False)

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
        dqn = DQN(buffer=BasicExperienceBuffer(size=10, batch_size=2), gamma=0.9, epsilon=0.1, policy=DiscreteRandomPolicy(num_actions=2), start_learning_step=1000, target_update_freq=1000, update_freq=1000, optimizer=None)
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
            dqn.experience_buffer.add(obs, action, reward, next_obs, terminated)
            # update the new obs
            obs = next_obs

        assert len(dqn.experience_buffer) > 2
        

        class Policy7:
            def __init__(self):
                self.num_actions = 7
            def __call__(self, obs):
                return 7

        dqn = DQN(buffer=BasicExperienceBuffer(size=10, batch_size=2), 
        gamma=0.9, epsilon=0.1, 
        policy=Policy7(), optimizer=None,
        start_learning_step=1000, target_update_freq=1000, update_freq=1000)
        
        total_7s = 0
        # test epsilon randomness is working
        for i in range(1000):
            action = dqn(torch.FloatTensor([-1]))
            if action == 7:
                total_7s+=1

        # should be a bit above 900 7s
        assert total_7s > 860 and total_7s < 960

        buffer = [Experience(0, 1, 1, 0, False), Experience(0, 0, 0, 0, False), Experience(0, 1, 1, 0, True)]
    
    def test_next_state_maxq_values(self):
        net = LinearNet()
        weights  = net.state_dict()
        weights['fc.weight'] = torch.FloatTensor([[2.0],[-1.0]])
        net.load_state_dict(weights)
        policy = MaxQPolicy(net, num_actions=2)
        # instantiate a DQN agent
        buffer = BasicExperienceBuffer(size=10, batch_size=2)
        dqn = DQN(buffer=BasicExperienceBuffer(size=10, batch_size=2), 
        gamma=0.9, epsilon=0.1, 
        policy=policy, optimizer=torch.optim.Adam(net.parameters(), lr=0.1),
        start_learning_step=1000, target_update_freq=1000, update_freq=1000)
        
        buffer.add(np.array([1.0]), 2, 3, np.array([4.0]), False)
        buffer.add(np.array([1.0]), 2, 3, np.array([-2.0]), False)
        buffer.add(np.array([1.0]), 2, 3, np.array([1.0]), True)

        states, actions, rewards, next_states, dones = unzip_experience_buffer(buffer)
        assert torch.equal(dqn.next_state_maxq_values(next_states, dones), torch.Tensor([8.0, 2.0, 0.0]))

    def test_compute_target_values(self):
        net = LinearNet()
        weights  = net.state_dict()
        weights['fc.weight'] = torch.FloatTensor([[2.0],[-1.0]])
        net.load_state_dict(weights)
        policy = MaxQPolicy(net, num_actions=2)
        # instantiate a DQN agent
        buffer = BasicExperienceBuffer(size=10, batch_size=2)
        dqn = DQN(buffer=BasicExperienceBuffer(size=10, batch_size=2), 
        gamma=0.9, epsilon=0.1, 
        policy=policy, optimizer=torch.optim.Adam(net.parameters(), lr=0.1),
        start_learning_step=1000, target_update_freq=1000, update_freq=1000)
        
        buffer.add(np.array([1.0]), 2, 3, np.array([4.0]), False)
        buffer.add(np.array([4.0]), 2, 3, np.array([-2.0]), False)
        buffer.add(np.array([-2.0]), 2, 3, np.array([1.0]), True)

        states, actions, rewards, next_states, dones = unzip_experience_buffer(buffer)

        target_values = dqn.compute_target_values(rewards, next_states, dones)

        assert torch.equal(target_values, torch.Tensor([3.0 + dqn.gamma * 8.0, (3.0 + dqn.gamma * 2.0), 3.0]))
        
    def test_compute_q_values(self):
        net = LinearNet()
        weights  = net.state_dict()
        weights['fc.weight'] = torch.FloatTensor([[2.0],[-1.0]])
        net.load_state_dict(weights)
        policy = MaxQPolicy(net, num_actions=2)
        # instantiate a DQN agent
        buffer = BasicExperienceBuffer(size=10, batch_size=2)
        dqn = DQN(buffer=BasicExperienceBuffer(size=10, batch_size=2), 
        gamma=0.9, epsilon=0.1, 
        policy=policy, optimizer=torch.optim.Adam(net.parameters(), lr=0.1),
        start_learning_step=1000, target_update_freq=1000, update_freq=1000)
        
        buffer.add(np.array([1.0]), 0, 3, np.array([4.0]), False)
        buffer.add(np.array([4.0]), 1, 3, np.array([-2.0]), False)
        buffer.add(np.array([-2.0]), 0, 3, np.array([1.0]), True)

        states, actions, rewards, next_states, dones = unzip_experience_buffer(buffer)

        target_values = dqn.compute_q_values(states, actions)

        assert torch.equal(target_values, torch.FloatTensor([2.0, -4.0, -4.0]))
        assert target_values.requires_grad == True

    def test_calc_loss(self):
        net = LinearNet()
        weights  = net.state_dict()
        weights['fc.weight'] = torch.FloatTensor([[2.0],[-1.0]])
        net.load_state_dict(weights)
        policy = MaxQPolicy(net, num_actions=2)
        # instantiate a DQN agent
        buffer = BasicExperienceBuffer(size=10, batch_size=3)
        dqn = DQN(buffer=buffer, 
        gamma=0.9, epsilon=0.1, 
        policy=policy, optimizer=torch.optim.Adam(net.parameters(), lr=0.1),
        start_learning_step=1000, target_update_freq=1000, update_freq=1000)
        
        buffer.add(np.array([1.0]), 0, 3, np.array([4.0]), False)
        buffer.add(np.array([4.0]), 1, 3, np.array([-2.0]), False)
        buffer.add(np.array([-2.0]), 0, 3, np.array([1.0]), True)

        sample = buffer.sample()

        loss = dqn.calc_loss(sample)

        q_values = [2, -4, -4]

        targets = [3 + dqn.gamma * 8.0, 3 + dqn.gamma * 2.0, 3]

        true_losses = [(q_values[0] - targets[0])**2, (q_values[1] - targets[1])**2, (q_values[2] - targets[2])**2]
        true_loss = sum(true_losses)/3

        assert loss.item() == approx(true_loss)
        assert loss.requires_grad == True

    def test_update(self):
        net = LinearNet()
        weights  = net.state_dict()
        weights['fc.weight'] = torch.FloatTensor([[2.0],[-1.0]])
        net.load_state_dict(weights)
        policy = MaxQPolicy(net, num_actions=2)
        # instantiate a DQN agent
        buffer = BasicExperienceBuffer(size=10, batch_size=1)
        dqn = DQN(buffer=buffer, 
        gamma=0.9, epsilon=0.1, 
        policy=policy, optimizer=torch.optim.Adam(net.parameters(), lr=0.001),
        start_learning_step=1000, target_update_freq=1000, update_freq=1000)
        
        buffer.add(np.array([1.0]), 0, 3, np.array([4.0]), False)

        sample = buffer.sample()
        dqn.calc_loss(sample).backward()
        
        # hand computed gradient for comparison
        assert all(torch.eq(next(dqn.policy.net.parameters()).grad.view(-1), torch.FloatTensor([2*(3+0.9*8 - 2)*-1, 0.0])))
        dqn.policy.net.zero_grad()

        dqn.update()

        assert dqn.policy.net.state_dict()['fc.weight'][0] > 2


    def test_train(self):
        net = LinearNet(4, 2)
        policy = MaxQPolicy(net, num_actions=2)
        # instantiate a DQN agent
        dqn = DQN(buffer=BasicExperienceBuffer(size=200, batch_size=20), 
        gamma=0.9, epsilon=0.1, 
        policy=policy, optimizer=torch.optim.Adam(net.parameters(), lr=0.1),
        start_learning_step=20, target_update_freq=4, update_freq=20)

        env = gym.make('CartPole-v1')

        train(dqn, env, 100)

