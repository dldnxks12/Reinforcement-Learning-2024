import torch
import torch.nn as nn

import numpy as np


def log_to_txt(env_name, total_step, result, index):

    f = open('./log-8/' + env_name + '-' + str(index) + '.txt', 'a')
    log = str(total_step) + ' ' + str(result) + '\n'
    f.write(log)
    f.close()

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers.
        Reference: https://github.com/MishaLaskin/rad/blob/master/curl_sac.py"""

    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)

def hard_update(network, target_network):
    for param, target_param in zip(network.parameters(), target_network.parameters()):
        target_param.data.copy_(param.data)

def soft_update(network, target_network, tau):
    for param, target_param in zip(network.parameters(), target_network.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def set_seed(random_seed):

    torch.manual_seed(777)
    np.random.seed(777)

    return 777

def make_env(env_name, seed):
    import os
    import gymnasium as gym

    xml_file = os.getcwd() + "/environment/assets/inverted_double_pendulum.xml"

    env = gym.make("InvertedDoublePendulum-v4", model_path=xml_file)
    env.action_space.seed(seed)

    env_eval = gym.make("InvertedDoublePendulum-v4", model_path=xml_file)
    env_eval.action_space.seed(seed)

    return env, env_eval