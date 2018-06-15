import numpy as np

USE_MANUAL_INPUT = False
USE_RANDOM_AGENT = False
USE_EPSILON_GREEDY = False
USE_THOMPSON_SAMPLING = True

np.random.seed(7)
N = 3  # Number of bandit arms.
num_steps = 100000
reward_received = 0.

bandit_probs = np.zeros(shape=[N], dtype=np.float32)
bandit_probs[0] = 0.2
bandit_probs[1] = 0.5
bandit_probs[2] = 0.7

S = np.zeros(shape=[N], dtype=np.int32)  # Number of successes so far.
F = np.zeros(shape=[N], dtype=np.int32)  # Number of failures so far.
probs = np.zeros(shape=[N], dtype=np.float32)

if USE_EPSILON_GREEDY:
    epsilon = 0.1
elif USE_THOMPSON_SAMPLING:
    alpha = 1.0; beta = 1.0

for step in range(num_steps):
    if USE_MANUAL_INPUT:
        action = int(input("\nType an action (integer between 0 and 2): "))
    elif USE_RANDOM_AGENT:
        action = np.random.randint(0, N)
    elif USE_EPSILON_GREEDY:
        for i in range(N):
            probs[i] = (S[i] + 1) / (S[i] + 1 + F[i] + 1)
        if np.random.uniform() < epsilon:
            action = np.random.randint(0, N)
        else:
            action = np.argmax(probs)
    elif USE_THOMPSON_SAMPLING:
        for i in range(N):
            probs[i] = np.random.beta(S[i] + alpha, F[i] + beta)
        action = np.argmax(probs)

    # Now that we've chosen one bandit arm (the action),
    # pull that arm and see if we get rewarded.
    reward = 0.
    p = np.random.uniform()
    if p < bandit_probs[action]:
        S[action] += 1  # Success
        reward += 1.
    else:
        F[action] += 1  # Failure
    reward_received += reward

    if USE_MANUAL_INPUT:
       print("Reward = {}".format(reward))

print("Mean reward received = {}".format(reward_received / num_steps))




