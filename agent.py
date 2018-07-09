from collections import deque
import numpy as np
import tensorflow as tf

class DeepQNetwork:
    def __init__(self, config):
        self.n_actions = config.n_actions
        self.replay_memory = deque([], maxlen=500000)

        self.discount_factor = config.discount_factor
        self.batch_size = config.batch_size
        self.e_min = config.e_min
        self.e_max = config.e_max
        self.e_decay_steps = 2000000

        self.checkpoint_path = config.checkpoint_path
        self.saver = tf.train.Saver()
        self.session = tf.Session()
        self.init = tf.global_variables_initializer()

        self.input_height = config.screen_height
        self.input_width = config.screen_width
        self.input_channels = config.input_channels


        self.step = 0

    def build_network(self):
        state = tf.placeholder(tf.float32, shape=[None, self.input_width, self.input_width, self.input_channels])

        conv_activation = tf.nn.relu
        hidden_activation = tf.nn.relu

        filters = [32, 64, 64]
        kernel_sizes = [8, 4, 3]
        strides = [4, 2, 1]
        paddings = ['SAME'] * 3
        initializer = tf.contrib.layers.variance_scaling_initializer()
        n_hidden = [512]

        conv1 = tf.layers.conv2d(state, filters=filters[0], kernel_size=kernel_sizes[0], strides=strides[0],
                                 padding=paddings[0], activation=conv_activation, initializer=initializer)
        conv2 = tf.layers.conv2d(conv1, filters=filters[1], kernel_size=kernel_sizes[1], strides=strides[1],
                                 padding=paddings[1], activation=conv_activation, initializer=initializer)
        conv3 = tf.layers.conv2d(conv2, filters=filters[2], kernel_size=kernel_sizes[2], strides=strides[2],
                                 padding=paddings[2], activation=conv_activation, initializer=initializer)

        conv3_shape = conv3.get_shape().as_list()
        conv3_flat = tf.reshape(conv3, shape=[-1, conv3_shape[1] * conv3_shape[2] * conv3_shape[3]])

        fc1 = tf.layers.dense(conv3_flat, n_hidden[0], activation=hidden_activation, initializer=initializer)
        outputs = tf.layers.dense(fc1, self.n_actions, kernel_initializer=initializer)
        return outputs

    def sample_memories(self):
        indices = np.random.permutation(len(self.replay_memory))[:self.batch_size]
        batch_memories = [[], [], [], [], []]
        for i in indices:
            memory = self.replay_memory[i]
            for category, value in zip(batch_memories, memory):
                category.append(value)
        batch_memories = [np.array(category) for category in batch_memories]
        return (batch_memories[0], batch_memories[1], batch_memories[2].reshape(-1, 1),
                batch_memories[3], batch_memories[4].reshape(-1, 1))

    def select_action(self, q_values):
        epsilon = max(self.e_min, self.e_max - (self.e_max - self.e_min) * self.step/self.e_decay_steps)
        if np.random.randn() < epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(q_values)




