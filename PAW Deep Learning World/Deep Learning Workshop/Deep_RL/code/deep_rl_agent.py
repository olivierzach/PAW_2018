import numpy as np
from copy import deepcopy
import tensorflow as tf

HIDDEN_LAYER_SIZE = 64
NUM_LSTM_UNITS = 128
LSTM_BACKPROP_WINDOW_LENGTH = 16

class DeepRLNetwork(object):
    def __init__(self, scope, observation_space_size, action_space_size):
        with tf.variable_scope(scope):
            # Configure the inputs. The [None] dimension is the time step. There is no batch dimension, except later to satisfy dynamic_rnn's API.
            self.observations = tf.placeholder(tf.float32, [None, observation_space_size])
            self.actions = tf.placeholder(tf.int32, [None])
            self.td_targets = tf.placeholder(tf.float32, [None])
            self.advantages = tf.placeholder(tf.float32, [None])

            observations_reshaped = tf.reshape(self.observations, [1, -1, observation_space_size])

            # Configure the LSTM.
            cell = tf.nn.rnn_cell.BasicLSTMCell(NUM_LSTM_UNITS, state_is_tuple=True)
            self.initial_rnn_state = tf.nn.rnn_cell.LSTMStateTuple(tf.placeholder(tf.float32, [1, NUM_LSTM_UNITS]), tf.placeholder(tf.float32, [1, NUM_LSTM_UNITS]))
            rnn_outputs, self.final_rnn_state = tf.nn.dynamic_rnn(cell, observations_reshaped, initial_state=self.initial_rnn_state)
            rnn_outputs_reshaped = tf.reshape(rnn_outputs, [-1, NUM_LSTM_UNITS])  # Remove the first (batch) dimension.

            # Calculate the state value (va).
            va_weights_1 = tf.Variable(tf.truncated_normal([NUM_LSTM_UNITS, HIDDEN_LAYER_SIZE], stddev=1.0 / tf.sqrt(1.0 * NUM_LSTM_UNITS)))
            va_bias_1 = tf.Variable(tf.constant(0., shape=[HIDDEN_LAYER_SIZE]))
            va_hidden_layer = tf.nn.relu(tf.matmul(rnn_outputs_reshaped, va_weights_1) + va_bias_1)
            va_weights_2 = tf.Variable(tf.truncated_normal([HIDDEN_LAYER_SIZE, 1], stddev=1.0 / tf.sqrt(1.0 * HIDDEN_LAYER_SIZE)))
            va_bias_2 = tf.Variable(tf.constant(0., shape=[1]))
            va_ = tf.matmul(va_hidden_layer, va_weights_2) + va_bias_2

            # Calculate the policy distribution (pi).
            pi_weights_1 = tf.Variable(tf.truncated_normal([NUM_LSTM_UNITS, HIDDEN_LAYER_SIZE], stddev=1.0 / tf.sqrt(1.0 * NUM_LSTM_UNITS)))
            pi_bias_1 = tf.Variable(tf.constant(0., shape=[HIDDEN_LAYER_SIZE]))
            pi_hidden_layer = tf.nn.relu(tf.matmul(rnn_outputs_reshaped, pi_weights_1) + pi_bias_1)
            pi_weights_2 = tf.Variable(tf.truncated_normal([HIDDEN_LAYER_SIZE, action_space_size], stddev=1.0 / tf.sqrt(1.0 * HIDDEN_LAYER_SIZE)))
            pi_bias_2 = tf.Variable(tf.constant(0., shape=[action_space_size]))
            pi_logits = tf.matmul(pi_hidden_layer, pi_weights_2) + pi_bias_2

            self.va = tf.reshape(va_, [-1])  # Remove the last dimension.
            self.pi_probs = tf.nn.softmax(pi_logits)

            if scope == 'global':
                # Things specific to the global network...
                self.saver = tf.train.Saver()  # Only used for loading models.
            else:
                # Things specific to the regular local networks...
                if scope == '0':
                    self.saver = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=1)  # Only used for saving models.

                # Calculate the state value's total loss over all elements of the sequence.
                va_losses = tf.squared_difference(self.va, self.td_targets)
                va_loss = tf.reduce_sum(va_losses)

                # Calculate the policy's total loss over all elements of the sequence.
                pi_clipped_probs = tf.clip_by_value(self.pi_probs, 1e-20, 1.0)
                actions_onehot = tf.one_hot(self.actions, action_space_size, dtype=tf.float32)
                chosen_action_probs = tf.reduce_sum(pi_clipped_probs * actions_onehot, [1])
                pi_losses = tf.log(chosen_action_probs) * self.advantages
                pi_loss = -tf.reduce_sum(pi_losses)
                entropy_term = -tf.reduce_sum(pi_clipped_probs * tf.log(pi_clipped_probs))

                # Get the gradients from the local network.
                total_loss = pi_loss + 0.5 * va_loss - entropy_term * 0.01
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                gradients = tf.gradients(total_loss, local_vars)

                # Clip the gradients to avoid explosions.
                clipped_gradients, grad_norms = tf.clip_by_global_norm(gradients, 10.)

                # Use an optimizer to apply the gradients to the global copy of the parameters.
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                optimizer = tf.train.AdamOptimizer(name='Adam_optimizer', learning_rate=0.001)
                self.modify_global_weights = optimizer.apply_gradients(zip(clipped_gradients, global_vars))

# Returns the TF ops for copying the trainable parameters (weights) from one scope to another.
def copy_weights(src_scope, tar_scope):
    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, src_scope)
    tar_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tar_scope)
    ops = []
    for src_var, tar_var in zip(src_vars, tar_vars):
        ops.append(tar_var.assign(src_var))
    return ops

class DeepRLAgent(object):
    def __init__(self, agent_name, observation_space_size, action_space_size, sess):
        self.name = agent_name
        self.observation_space_size = observation_space_size
        self.action_space_size = action_space_size
        self.sess = sess
        if agent_name == '0':
            self.global_net = DeepRLNetwork('global', self.observation_space_size, action_space_size)
        self.local_net = DeepRLNetwork(self.name, self.observation_space_size, action_space_size)
        self.update_local_weights = copy_weights('global', self.name)  # For copying global weights to local weights.
        self.reward_array = np.array([0.] * (LSTM_BACKPROP_WINDOW_LENGTH))
        self.state_value_array = np.array([0.] * (LSTM_BACKPROP_WINDOW_LENGTH + 1))  # The extra element is for the next_state_value.
        self.last_observation = np.zeros(self.observation_space_size)
        self.state_value = 0.
        self.zero_state_value = np.array([0.])

    def reset_state(self):
        self.num_training_frames_in_buffer = 0
        self.observation_list = []
        self.action_list = []
        self.rnn_state = tf.nn.rnn_cell.LSTMStateTuple(np.zeros([1, NUM_LSTM_UNITS]), np.zeros([1, NUM_LSTM_UNITS]))
        self.cached_rnn_state = deepcopy(self.rnn_state)
        return 0

    def copy_global_weights(self):
        self.sess.run(self.update_local_weights)

    def save_local_model(self, model_path):
        self.local_net.saver.save(self.sess, model_path)

    def load_model(self, model_path):
        self.global_net.saver.restore(self.sess, model_path)

    def step(self, prev_action, prev_reward, observation):
        assert(prev_action >= 0)

        self.last_observation[0:self.observation_space_size] = observation[0:self.observation_space_size]

        # Run one step forward in time (alternating with the environment).
        self.output_probs, self.state_value, self.rnn_state = self.sess.run(
            [self.local_net.pi_probs, self.local_net.va, self.local_net.final_rnn_state],
            {self.local_net.observations: [self.last_observation],
             self.local_net.initial_rnn_state: self.rnn_state})

        # Select one action.
        action = np.random.choice(self.action_space_size, p=self.output_probs[0])

        return action

    def adapt(self, action, reward, done):
        # Buffer one frame of data for eventual training.
        self.observation_list.append(np.copy(self.last_observation))  # Forced deep copy.
        self.action_list.append(action)
        self.state_value_array[self.num_training_frames_in_buffer] = self.state_value
        self.reward_array[self.num_training_frames_in_buffer] = reward
        self.num_training_frames_in_buffer += 1

        if done:
            self.adapt_on_end_of_episode()
        elif self.num_training_frames_in_buffer == LSTM_BACKPROP_WINDOW_LENGTH:
            self.adapt_on_end_of_sequence()

    def adapt_on_end_of_episode(self):
        # Train with a next state value of zero, because there aren't any rewards after the end of the episode.
        self.train(self.zero_state_value)

    def adapt_on_end_of_sequence(self):
        # Peek at the state value of the next observation, for TD calculation.
        next_state_value = self.sess.run(self.local_net.va,
            {self.local_net.observations: [self.last_observation],
             self.local_net.initial_rnn_state: self.rnn_state})

        # Train.
        self.train(next_state_value)

        # Reset things for the next adaptation.
        self.num_training_frames_in_buffer = 0
        self.observation_list = []
        self.action_list = []
        self.cached_rnn_state = deepcopy(self.rnn_state)

    def train(self, next_state_value):
        # Populate the td_target array (for value update) and the advantage array (for policy update).
        # Note that value errors are backpropagated through the current value calculation for value updates, but not for policy updates.
        td_target_list = []
        advantage_list = []
        self.state_value_array[self.num_training_frames_in_buffer] = next_state_value

        # N-step lookahead.
        td_target = next_state_value[0]
        for i in range(self.num_training_frames_in_buffer - 1, -1, -1):
            td_target = self.reward_array[i] + 0.9 * td_target
            advantage = td_target - self.state_value_array[i]
            td_target_list.append(td_target)
            advantage_list.append(advantage)
        td_target_list.reverse()
        advantage_list.reverse()

        # Perform the training.
        _ = self.sess.run(self.local_net.modify_global_weights,
            {self.local_net.observations: np.array(self.observation_list),
             self.local_net.actions: np.array(self.action_list),
             self.local_net.td_targets: np.array(td_target_list),
             self.local_net.advantages: np.array(advantage_list),
             self.local_net.initial_rnn_state: self.cached_rnn_state})

        # Get a local copy of the updated global weights.
        self.copy_global_weights()
