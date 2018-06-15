import argparse
import random
import numpy as np

SEED = 25912
np.random.seed(SEED)
random.seed(SEED)

# Parse the command line arguments.
parser = argparse.ArgumentParser('default arg parser')
parser.add_argument('--train', action='store_true', default=False, help='Whether to train a model.')
parser.add_argument('--display', action='store_true', default=False, help='Whether to display the game.')
args = parser.parse_args()
train = args.train
display = args.display

# Launch one worker, which creates the agent and the environment.
from worker import Worker
worker = Worker(train, display)
worker.run()
