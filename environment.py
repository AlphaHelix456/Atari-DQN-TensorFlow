import gym
import random


class AtariEnvironment:
    def __init__(self, config):
        self.env = gym.make(config.env)
        self.n_actions = self.env.action_space.n
        self.lives = self.env.unwrapped.ale.lives()
        self.skip_steps = config.skip_steps
        self.render = config.render
        self.screen_width = config.screen_width
        self.screen_height = config.screen_height

    def new_game(self):
        obs = self.env.reset()
        self.lives = self.env.unwrapped.ale.lives()
        return obs

    def new_game_with_random_skip_start(self):
        obs = self.env.reset()
        random_skip_steps = random.randint(1, self.skip_steps+1)

        for _ in range(random_skip_steps):
            obs, reward, done, _ = self.env.step(0)
            if self.render:
                self.env.render()

        self.lives = self.env.unwrapped.ale.lives()
        return obs

    def new_game_with_max_skip_start(self):
        obs = self.env.reset()

        for _ in range(self.skip_steps):
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



