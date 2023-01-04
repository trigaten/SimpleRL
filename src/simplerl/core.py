from abc import ABC, abstractmethod
from collections import namedtuple
from simplerl.hooks import Hook

"""A `Policy` takes in an observation and returns an action or array of 
log probs for actions."""
class Policy(ABC):
    
    @abstractmethod
    def __call__(x):
        raise Exception("Not Implemented")

"""An `Agent` is an object which contains 0 or more policies and is 
responsible for training them."""
class Agent(ABC):
    
    @abstractmethod
    def __call__(x):
        raise Exception("Not Implemented")

    def pre_experiment_stage(self):
        pass
    
    def post_episode_stage(self, episode):
        pass

    def pre_act_stage(self):
        pass

    def post_act_stage(self):
        pass

    def pre_episode_stage(self, episode):
        pass

def train(agent, env, episodes, hook=Hook()):
    # "global" state of training
    training_done = False
    # how many episodes have been completed thus far
    episodes_complete = 0

    agent.pre_experiment_stage()

    while not training_done:
        obs = env.reset()
        done = False

        hook.pre_episode_stage(episodes_complete)
        agent.pre_episode_stage(episodes_complete)

        while not done:
            hook.pre_act_stage()
            agent.pre_act_stage()
            
            action = agent(obs)

            next_obs, reward, done, info = env.step(action)
            
            agent.post_act_stage()
            hook.post_act_stage(reward)

            agent.post_experience(obs, action, reward, next_obs, done)

        episodes_complete+= 1

        hook.post_episode_stage(episodes_complete)
        agent.post_episode_stage(episodes_complete)

        if episodes_complete > episodes:
            training_done = True



Experience = namedtuple("Experience", "s a r s_ done")
