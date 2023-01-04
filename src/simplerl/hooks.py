from torch.utils.tensorboard import SummaryWriter
import torch

class Hook:
    def pre_experiment_stage(self):
        pass
    
    def post_episode_stage(self, episode):
        pass

    def pre_act_stage(self):
        pass

    def post_act_stage(self, reward):
        pass

    def pre_episode_stage(self, episode):
        pass

class RewardLoggerHook(Hook):
    def __init__(self, log_path):
        self.logger = SummaryWriter(log_path)
        self.episode = 0
        self.episode_reward = 0

    def post_act_stage(self, reward):
        self.episode_reward+= reward

    def post_episode_stage(self, episode):
        self.logger.add_scalar('reward', self.episode_reward, self.episode)
        # print(f"Episode {self.episode} reward: {self.episode_reward}")
        self.episode_reward = 0
        self.episode+=1

class TestPerformanceLogger(Hook):
    def __init__(self, env, agent, episode_frequency, log_path):
        self.logger = SummaryWriter(log_path)
        self.env = env
        self.agent = agent
        self.episode_frequency = episode_frequency

    def post_episode_stage(self, episode):
        if episode % self.episode_frequency == 0:
            env = self.env
            agent = self.agent
            totals = []
            for i in range(4):
                obs, info = env.reset()
                done = False
                total_reward = 0
                while not done:
                    action = agent.policy(torch.FloatTensor([obs]))
                    obs, reward, truncated, terminated, info = env.step(action)
                    done = truncated or terminated
                    total_reward += reward
                totals.append(total_reward)

            self.logger.add_scalar('test_reward', sum(totals)/len(totals), episode)

class ComposedHook(Hook):
    def __init__(self, hooks):
        self.hooks = hooks

    def pre_experiment_stage(self):
        for hook in self.hooks:
            hook.pre_experiment_stage()

    def post_episode_stage(self, episode):
        for hook in self.hooks:
            hook.post_episode_stage(episode)

    def pre_act_stage(self):
        for hook in self.hooks:
            hook.pre_act_stage()

    def post_act_stage(self, reward):
        for hook in self.hooks:
            hook.post_act_stage(reward)

    def pre_episode_stage(self, episode):
        for hook in self.hooks:
            hook.pre_episode_stage(episode)