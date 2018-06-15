import numpy as np

class RandomAgent(object):
    def __init__(self, action_space_size):
        self.action_space_size = action_space_size

    def reset_state(self):
        return 0

    def load_model(self, model_path):
        pass

    def copy_global_weights(self):
        pass

    def save_local_model(self, model_path):
        pass

    def step(self, prev_action, prev_reward, observation):
        return np.random.choice(self.action_space_size)

    def adapt(self, action, reward, done):
        pass

    def adapt_on_end_of_episode(self):
        pass

    def train(self, next_state_value):
        pass
