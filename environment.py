import gym


class Environment:
    def __init__(self, config):
        self.env = gym.make(config.env)
        self.action_repeat = config.action_repeat
        self.n_actions = self.env.action_space.n
        self.render = config.render
        self.screen_height = config.screen_height
        self.screen_width = config.screen_width

    def preprocess(self, obs):
        img = obs[1:176:2, ::2]  # Crop and downsize
        img = img.mean(axis=2)  # Convert to grayscale
        img = (img - 128) / 128 - 1  # normalize from -1 to 1
        return img.reshape(self.screen_height, self.screen_width, 1)
