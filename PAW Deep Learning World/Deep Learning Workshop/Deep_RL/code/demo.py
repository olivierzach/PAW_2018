import argparse
import threading
import multiprocessing
import random
import numpy as np
import tensorflow as tf
from worker import NUM_WORKERS

SEED = 25912

# Parse the command line arguments.
parser = argparse.ArgumentParser('default arg parser')
parser.add_argument('--train', action='store_true', default=False, help='Whether to train a model.')
parser.add_argument('--display', action='store_true', default=False, help='Whether to display the game.')
parser.add_argument('--model', action='store_true', default=False, help='Whether to load the last trained model.')
args = parser.parse_args()
train = args.train
display = args.display
model = args.model

# Initialize some basic things.
if display:
    NUM_WORKERS = 1
num_cpus = multiprocessing.cpu_count()
num_cores = int(num_cpus / 2)
config = tf.ConfigProto(device_count={"CPU": num_cores}, intra_op_parallelism_threads=1)
config.graph_options.optimizer_options.opt_level = -1
sess = tf.Session(config=config)
tf.set_random_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# For each thread, create one worker, which will instantiate one agent-environment pair.
# This is where an agent may create a TF network.
coord = tf.train.Coordinator()
threads = []
from worker import Worker
workers = []
environments = []
for i in range(NUM_WORKERS):
    with tf.device("/cpu:{}".format(i % num_cores)):
        worker = Worker(i, NUM_WORKERS, sess, train, display, model)
        workers.append(worker)
        environments.append(worker.environment)
workers[0].all_environments = environments  # The first worker is the reporter, so it will want to collect results from the others.

# Initialize all the TF networks.
sess.run(tf.global_variables_initializer())

# Load the model, if needed.
if model:
    workers[0].load_model()

# Run the threads in parallel.
for worker in workers:
    t = threading.Thread(target=(lambda: worker.run()))
    threads.append(t)
    t.daemon = True
    t.start()
coord.join(threads)
