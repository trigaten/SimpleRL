import numpy as np

def discount_rewards(rewards, gamma):
    """
    Compute discounted rewards.
    :param rewards: list of rewards
    :param gamma: discount factor
    :return: list of discounted rewards
    """
    num_rewards = len(rewards)
    # set last discounted reward as last reward
    # this reduces branching
    discounted_rewards = np.zeros_like(rewards)
    discounted_rewards[num_rewards-1] = rewards[num_rewards-1]
    # reverse iterate over rewards
    for t in reversed(range(0, num_rewards-1)):
        # compute discounted reward as reward at current time step plus
        # discounted sum of discounted_rewards from next time step
        discounted_rewards[t] = rewards[t] + gamma * discounted_rewards[t+1]
    return discounted_rewards

def compare_lists_of_np_arrays(l1, l2):
    """Compare two lists of numpy arrays."""
    equal = True
    for i in range(len(l1)):
        equal = np.array_equal(l1[i], l2[i]) & equal

    return equal

def unzip_experience_buffer(buffer):
    """Unzip experience buffer into separate lists of observations, actions, 
    rewards, and next observations."""
    obs, actions, rewards, next_obs, dones = [], [], [], [], []
    for s, a, r, s_, done in buffer:
        obs.append(s)
        actions.append(a)
        rewards.append(r)
        next_obs.append(s_)
        dones.append(done)
        
    return obs, actions, rewards, next_obs, dones

