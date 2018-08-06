import numpy as np
import os
import tensorflow as tf
from experience import ReplayMemory
from summary import Summary
from tqdm import tqdm


class DeepQAgent:
    def __init__(self, env, sess, config):
        self.env = env
        self.sess = sess

        self.discount_factor = config.discount_factor
        self.batch_size = config.batch_size
        self.eps_min = config.eps_min
        self.eps_max = config.eps_max
        self.eps_decay_steps = config.eps_decay_steps
        self.screen_height = config.screen_height
        self.screen_width = config.screen_width

        if config.random_start:
            self.new_game = self.env.new_random_game
        else:
            self.new_game = self.env.new_game

        self.to_train = config.to_train
        self.max_train_steps = config.max_train_steps
        self.train_freq = config.train_freq
        self.save_freq = config.save_freq
        self.copy_freq = config.copy_freq

        if not os.path.isdir(config.checkpoint_dir):
            os.mkdir(config.checkpoint_dir)

        self.summary = Summary(config.test_freq, self.sess, config.checkpoint_dir)
        self.checkpoint_path = os.path.join(config.checkpoint_dir, 'ckpt')

        self.replay_memory = ReplayMemory(config.replay_memory_size, config.batch_size)

        self.step = tf.Variable(0, trainable=False, name='global_step')

        self.inputs = tf.placeholder(tf.float32, shape=[None, self.screen_height, self.screen_width, 1])
        self.actions = tf.placeholder(tf.int32, shape=[None], name='actions')
        self.targets = tf.placeholder(tf.float32, shape=[None, 1], name='target_q_values')

        self.online_q_values, online_vars = self.build_network(self.inputs, 'online_q_network')
        self.target_q_values, target_vars = self.build_network(self.inputs, 'target_q_network')

        copy_ops = [target_var.assign(online_vars[var_name]) for var_name, target_var in target_vars.items()]
        self.copy_online_to_target = tf.group(*copy_ops)

        q_values = tf.reduce_mean(self.target_q_values * tf.one_hot(self.actions, self.env.n_actions),
                                  axis=1, keepdims=True)

        error = tf.abs(self.targets - q_values)
        clipped_error = tf.clip_by_value(error, 0.0, 1.0)
        linear_error = 2 * (error - clipped_error)
        self.loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)

        optimizer = tf.train.RMSPropOptimizer(learning_rate=config.lr,
                                              decay=config.decay,
                                              momentum=config.momentum,
                                              epsilon=config.eps)
        self.train_op = optimizer.minimize(self.loss, global_step=self.step)

        self.saver = tf.train.Saver()
        self.init = tf.global_variables_initializer()

    def train(self):
        if os.path.isfile(self.checkpoint_path + '.index'):
            self.saver.restore(self.sess, self.checkpoint_path)
        else:
            self.init.run()
            self.copy_online_to_target.run()

        start_step = self.step.eval()
        done = True
        state = None
        iteration = 0
        progress_bar = tqdm(range(0, self.max_train_steps), initial=start_step, unit='steps')

        while True:

            iteration += 1
            if done:
                obs = self.new_game()
                state = self.env.preprocess(obs)

            eps = max(self.eps_min, self.eps_max - (self.eps_min - self.eps_max) * self.step.eval()/self.eps_decay_steps)

            q_values = self.online_q_values.eval(feed_dict={self.inputs: [state]})
            action = self.predict(eps, q_values)

            obs, reward, done, _ = self.env.step(action, self.to_train)
            next_state = self.env.preprocess(obs)

            self.replay_memory.add(state, action, reward, next_state, done)
            state = next_state

            if iteration % self.train_freq != 0:
                continue

            states, actions, rewards, next_states, continues = self.replay_memory.sample_memories()
            next_q_values = self.target_q_values.eval(feed_dict={self.inputs: next_states})
            max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
            y_values = rewards + continues * self.discount_factor * max_next_q_values
            loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={
                self.inputs: states, self.actions: actions, self.targets: y_values})
            step = self.step.eval()

            if step % self.copy_freq == 0:
                self.copy_online_to_target.run()

            if step % self.save_freq == 0:
                self.saver.save(self.sess, self.checkpoint_path)

            self.summary.record_step(step, action, reward, done, eps, np.max(q_values), loss)
            progress_bar.update(1)

            if step >= self.max_train_steps:
                break

    def play(self):
        pass

    def build_network(self, x, name):
        w_conv1 = self.weights([8, 8, 1, 32])
        b_conv1 = self.bias(32)

        w_conv2 = self.weights([4, 4, 32, 64])
        b_conv2 = self.bias(64)

        w_conv3 = self.weights([3, 3, 64, 64])
        b_conv3 = self.bias(64)

        conv1 = tf.nn.relu(self.conv2d(x, w_conv1, 4) + b_conv1)
        conv2 = tf.nn.relu(self.conv2d(conv1, w_conv2, 2) + b_conv2)
        conv3 = tf.nn.relu(self.conv2d(conv2, w_conv3, 1) + b_conv3)

        _, h, w, c = conv3.get_shape().as_list()
        conv3_flat = tf.reshape(conv3, shape=[-1, h * w * c])

        w_fc1 = self.weights([h * w * c, 512])
        b_fc1 = self.bias(512)

        w_fc2 = self.weights([512, self.env.n_actions])
        b_fc2 = self.bias(self.env.n_actions)

        fc1 = tf.nn.relu(tf.matmul(conv3_flat, w_fc1) + b_fc1)
        outputs = tf.matmul(fc1, w_fc2) + b_fc2

        trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        trainable_variables_by_name = {var.name[len(name):]: var
                                       for var in trainable_variables}
        return outputs, trainable_variables_by_name

    def predict(self, eps, q_values):
        if np.random.randn() < eps:
            return np.random.randint(self.env.n_actions)
        return np.argmax(q_values)

    def weights(self, kernel_shape):
        return tf.Variable(tf.truncated_normal(kernel_shape, stddev=0.01), dtype=tf.float32, trainable=self.to_train,
                           name='weights')

    def bias(self, output_dim):
        return tf.Variable(tf.constant(0.01, shape=[output_dim]), dtype=tf.float32, trainable=self.to_train,
                           name='biases')

    def conv2d(self, x, w, stride):
        return tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='SAME')
