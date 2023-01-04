from torch.utils.tensorboard import SummaryWriter

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
