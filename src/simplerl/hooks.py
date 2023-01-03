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
        self.rewards = []

    def post_act_stage(self, reward):
        self.rewards.append(reward)

    def post_episode_stage(self, episode):
        self.episode+=1

        self.logger.add_scalar('reward', sum(self.episode.reward), self.episode)

        self.rewards.clear()