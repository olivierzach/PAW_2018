import numpy as np

class RandomAgent(object):
    def __init__(self, observation_space_size, action_space_size):
        self.action_space_size = action_space_size

    def step(self, prev_reward, observation):
        return np.random.choice(self.action_space_size)
