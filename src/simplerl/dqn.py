# from core import Agent, Policy

# class DQN(Agent):
#     def __init__(self, buffer_size:int, batch_size:int, gamma:float, epsilon:float):
#         self.buffer_size = buffer_size
#         self.batch_size = batch_size
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.experience_buffer = []
#         self.policy = Policy()

#     def __call__(self, obs):

def dqn_train(env, agent):
    # "global" state of training
    training_done = False
    # how many episodes have been completed thus far
    episodes_complete = 0
    while not training_done:
        obs = env.reset()
        done = False
        while not done:
            obs, reward, done, info = agent(obs)

        episodes_complete+= 1
        if episodes_complete > 1000:
            training_done = True