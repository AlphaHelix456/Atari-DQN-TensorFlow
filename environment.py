import gym
import random


class AtariEnvironment:
    def __init__(self, config):
        self.env = gym.make(config.env)
        self.n_actions = self.env.action_space.n
        self.lives = self.env.unwrapped.ale.lives()
        self.max_random_start = config.max_random_start

        self.render = config.render
        self.screen_width = config.screen_width
        self.screen_height = config.screen_height

    def new_game(self):
        obs = self.env.reset()
        self.lives = self.env.unwrapped.ale.lives()
        return obs

    def new_random_game(self):
        obs = self.env.reset()

        for _ in range(random.randint(1, self.max_random_start+1)):
            obs, reward, done, _ = self.env.step(0)
            if self.render:
                self.env.render()

        self.lives = self.env.unwrapped.ale.lives()
        return obs

    def step(self, action, to_train):
        obs, reward, done, _ = self.env.step(action)
        #current_lives = self.env.unwrapped.ale.lives()

        #if to_train and self.lives > current_lives:
            #done = True

        if self.render:
            self.env.render()

        #if not done:
            #self.lives = current_lives

        return obs, reward, done

    def preprocess(self, obs):
        img = obs[1:176:2, ::2]  # Crop and downsize
        img = img.mean(axis=2)  # Convert to grayscale
        img = (img - 128) / 128 - 1  # normalize from -1 to 1
        return img.reshape(self.screen_height, self.screen_width, 1)



