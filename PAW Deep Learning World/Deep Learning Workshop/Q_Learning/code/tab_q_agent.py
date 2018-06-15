import numpy as np
import random

epsilon = 0.5
alpha = 0.3
gamma = 0.95

class TabQAgent(object):
    def __init__(self, observation_space_size, action_space_size):
        self.observation_space_size = observation_space_size  # The number of states in the MDP.
        self.action_space_size = action_space_size
        self.q = np.zeros(shape=[observation_space_size,action_space_size], dtype=np.float32)
        self.best_act = 0
        self.taken_act = 0
        self.best_q = 0.
        self.taken_q = 0.
        self.state = 0
        self.prev_taken_act = 0
        self.prev_state = 0

    def step(self, reward, observation):
        # Cache the previous step's info.
        self.prev_taken_act = self.taken_act
        self.prev_state = self.state

        self.state = observation

        # Find the best action to take.
        self.best_act = -1
        for action in range(self.action_space_size):
            q = self.q[self.state, action]
            if (self.best_act == -1) or (self.best_q < q):
                self.best_act = action
                self.best_q = q

        # Update Q.
        self.q[self.prev_state, self.prev_taken_act] += \
            alpha * (reward + gamma * self.best_q - self.q[self.prev_state, self.prev_taken_act])

        # Choose the action to take, based on epsilon-greedy.
        if random.random() < epsilon:
            # Explore
            self.taken_act = np.random.choice(self.action_space_size)
        else:
            self.taken_act = self.best_act
        self.taken_q = self.q[self.state, self.taken_act]

        return self.taken_act
