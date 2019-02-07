import tensorflow as tf
import numpy as np


class Summary:
    def __init__(self, test_freq, sess, checkpoint_dir):
        self.test_freq = test_freq
        self.sess = sess
        self.writer = tf.summary.FileWriter(checkpoint_dir + '/train', self.sess.graph)

        self.reset()

        with tf.variable_scope('summary'):
            scalar_summary_tags = [
                'average/reward', 'average/loss', 'average/q',
                'episode/max_reward', 'episode/min_reward', 'episode/avg_reward',
                'episode/num_game', 'training/epsilon']

            self.summary_placeholders = {}
            self.summary_ops = {}

            for tag in scalar_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder(tf.float32, None)
                self.summary_ops[tag] = tf.summary.scalar(tag, self.summary_placeholders[tag])

            histogram_summary_tags = ['episode/rewards', 'episode/actions']

            for tag in histogram_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder(tf.float32, None)
                self.summary_ops[tag] = tf.summary.histogram(tag, self.summary_placeholders[tag])

    def reset(self):
        self.num_game = 0
        self.num_actions = 0
        self.ep_reward = 0
        self.total_loss = 0
        self.total_reward = 0
        self.total_q = 0
        self.actions = []
        self.ep_rewards = []

    def record_step(self, step, action, reward, done, eps, q, loss):
        self.total_loss += loss
        self.total_reward += reward
        self.total_q += q
        self.actions.append(action)
        self.num_actions += 1

        if done:
            self.num_game += 1
            self.ep_rewards.append(self.ep_reward)
            self.ep_reward = 0
        else:
            self.ep_reward += reward

        if step % self.test_freq == 0:
            avg_q = self.total_q / self.num_actions
            avg_loss = self.total_loss / self.num_actions
            avg_reward = self.total_reward / self.num_actions

            if self.ep_rewards:
                max_ep_reward = np.max(self.ep_rewards)
                min_ep_reward = np.min(self.ep_rewards)
                avg_ep_reward = np.mean(self.ep_rewards)
            else:
                max_ep_reward = 0
                min_ep_reward = 0
                avg_ep_reward = 0

            self.write_summary({
                'average/q': avg_q,
                'average/loss': avg_loss,
                'average/reward': avg_reward,
                'episode/max_reward': max_ep_reward,
                'episode/min_reward': min_ep_reward,
                'episode/avg_reward': avg_ep_reward,
                'episode/num_game': self.num_game,
                'episode/actions': self.actions,
                'episode/rewards': self.ep_rewards,
                'training/epsilon': eps
            }, step)

            #print('summary: Avg loss {} Avg q {} Avg reward {} max reward {} min reward {} num game {}'.format(
              #avg_loss, avg_q, avg_reward, max_ep_reward, min_ep_reward, self.num_game))
            self.reset()

    def write_summary(self, tag_dict, step):
        summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict], {
            self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
        })

        for summary_str in summary_str_lists:
            self.writer.add_summary(summary_str, step)
