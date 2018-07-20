import gym
import random


class AtariEnvironment:
    def __init__(self, config):
        self.env = gym.make(config.env)
        self.n_actions = self.env.action_space.n
        self.lives = self.env.unwrapped.ale.lives()
        self.n_action_repeat = config.n_action_repeat
        self.max_random_start = config.max_random_start

        self.render = config.render
        self.screen_width = config.screen_width
        self.screen_height = config.screen_height
        self.use_cumulative_reward = config.use_cumulative_reward

    def new_game(self):
        obs = self.env.reset()
        self.lives = self.env.unwrapped.ale.lives()
        return obs

    def new_random_game(self):
        obs = self.env.reset()

        for i in range(random.randint(1, self.max_random_start)):
            obs, reward, done, _ = self.env.step(0)

        if self.render:
            self.env.render()

        self.lives = self.env.unwrapped.ale.lives()
        return obs

    def step(self, action, to_train):
        cumulative_reward = 0
        obs, reward, done, current_lives = [None] * 4
        for _ in range(self.n_action_repeat):
            obs, reward, done, _ = self.env.step(action)
            cumulative_reward += reward
            current_lives = self.env.unwrapped.ale.lives()

            if to_train and self.lives > current_lives:
                done = True
                break

        if self.render:
            self.env.render()

        if not done:
            self.lives = current_lives

        if self.use_cumulative_reward:
            return obs, cumulative_reward, done, {}
        else:
            return obs, reward, done, {}

    def preprocess(self, obs):
        img = obs[1:176:2, ::2]  # Crop and downsize
        img = img.mean(axis=2)  # Convert to grayscale
        img = (img - 128) / 128 - 1  # normalize from -1 to 1
        return img.reshape(self.screen_height, self.screen_width, 1)



