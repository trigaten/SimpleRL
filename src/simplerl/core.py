from abc import ABC, abstractmethod
from collections import namedtuple

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

def train(agent, env, episodes):
    # "global" state of training
    training_done = False
    # how many episodes have been completed thus far
    episodes_complete = 0
    while not training_done:
        obs = env.reset()
        done = False
        while not done:
            action = agent(obs)
            next_obs, reward, done, info = env.step(action)
            

        episodes_complete+= 1
        agent.post_stage("POST_EPISODE_STAGE", episodes_complete)
        
        if episodes_complete > episodes:
            training_done = True



Experience = namedtuple("Experience", "s a r s_ done")
