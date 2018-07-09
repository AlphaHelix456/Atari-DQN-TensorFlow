from __future__ import print_function
import tensorflow as tf


flags = tf.app.flags

# Environment
flags.DEFINE_string('env', 'MsPacman-v0', 'The Atari environment to be used')
flags.DEFINE_integer('action_repeat', 1, 'The number of actions to repeat')

# Agent
flags.DEFINE_boolean('dueling_q', False, 'Whether to use dueling deep q-network')
flags.DEFINE_boolean('double_q', False, 'Whether to use double q-learning')

# Training
flags.DEFINE_boolean('train', True, 'Whether to train or test')
flags.DEFINE_float('lr', 0.001, 'The learning rate')
flags.DEFINE_integer('batch_size', 32, 'The batch size for training')
flags.DEFINE_float('discount_factor', 0.99, 'The discount factor')

# Etc
flags.DEFINE_boolean('render', False, 'Whether to display the game screen')
flags.DEFINE_boolean('use_gpu', True, 'Whether to use GPU')
flags.DEFINE_integer('seed', 42, 'Value of random seed')

FLAGS = flags.FLAGS

tf.set_random_seed(FLAGS.seed)

if __name__ == '__main__':
    tf.app.run()