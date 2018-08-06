
class Config:
    # Environment
    env = 'MsPacMan-v0'
    screen_height = 88
    screen_width = 80
    n_action_repeat = 1
    random_start = True
    max_random_start = 30
    use_cumulative_reward = False
    render = False

    # Agent
    batch_size = 32
    discount_factor = 0.99
    eps_min = 0.1
    eps_max = 1.0
    eps_decay_steps = 2000000
    lr = 0.001
    momentum = 0.0
    decay = 0.90
    eps = 0.01
    max_train_steps = 4000000
    train_freq = 4
    save_freq = 1000
    copy_freq = 10000
    test_freq = 10000
    replay_memory_size = 500000

    # Other
    seed = 42
    to_train = True
    use_gpu = True
    gpu_fraction = '1/1'
    checkpoint_dir = ''


def get_config(flags):
    config = Config
    for k, v in flags.flag_values_dict().items():
        if hasattr(config, k):
            setattr(config, k, v)
    return config
