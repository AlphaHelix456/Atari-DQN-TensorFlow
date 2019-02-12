import os
import random
import tensorflow as tf
from agent import DeepQAgent
from environment import AtariEnvironment
from config import get_config


flags = tf.app.flags

# Environment
flags.DEFINE_string('env', 'MsPacman-v0', 'The name of the Atari environment to be used')
flags.DEFINE_integer('screen_width', 80, 'The width of the screen')
flags.DEFINE_integer('screen_height', 88, 'The height of the screen')
flags.DEFINE_string('skip_start', 'max', 'The method for skipping steps at start of new game')
flags.DEFINE_integer('skip_steps', 90, 'The max number of steps to skip at start of new game')
flags.DEFINE_boolean('done_after_life_lost', False, 'Whether to continue an episode after losing a life during training')

# Training
flags.DEFINE_boolean('to_train', True, 'Whether to train or test')
flags.DEFINE_float('lr', 0.001, 'The learning rate')
flags.DEFINE_integer('batch_size', 32, 'The batch size')
flags.DEFINE_float('discount_factor', 0.99, 'The discount factor')

# Other
flags.DEFINE_boolean('render', False, 'Whether to display the game screen')
flags.DEFINE_boolean('use_gpu', True, 'Whether to use GPU')
flags.DEFINE_string('gpu_fraction', '1/1', 'idx / # of gpu fraction e.g. 1/3, 2/3, 3/3')
flags.DEFINE_integer('seed', 42, 'Value of random seed')
flags.DEFINE_boolean('allow_soft_placement', True, 'Whether soft placement is allowed')
flags.DEFINE_string('checkpoint_dir', '', 'The absolute path to the checkpoint directory')

FLAGS = flags.FLAGS

tf.set_random_seed(FLAGS.seed)
random.seed(FLAGS.seed)


def calc_gpu_fraction(fraction_string):
    idx, num = fraction_string.split('/')
    idx, num = float(idx), float(num)

    fraction = idx / num
    return fraction


def main(_):
    if not tf.test.is_gpu_available() and FLAGS.use_gpu:
        raise Exception("use_gpu flag is true when no GPUs are available")

    assert FLAGS.checkpoint_dir != '', 'Checkpoint directory must be specified'

    if not FLAGS.to_train and not os.path.isfile(os.path.join(FLAGS.checkpoint_dir, 'ckpt.index')):
        raise Exception("Checkpoint directory must contain a trained model to do testing")

    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=calc_gpu_fraction(FLAGS.gpu_fraction), allow_growth=True)

    sess_config = tf.ConfigProto(
        log_device_placement=False,
        allow_soft_placement=FLAGS.allow_soft_placement,
        gpu_options=gpu_options
    )

    with tf.Session(config=sess_config) as sess:
        config = get_config(FLAGS)

        env = AtariEnvironment(config)

        agent = DeepQAgent(env, sess, config)

        if config.to_train:
            agent.train()
        else:
            agent.play()


if __name__ == '__main__':
    tf.app.run()
