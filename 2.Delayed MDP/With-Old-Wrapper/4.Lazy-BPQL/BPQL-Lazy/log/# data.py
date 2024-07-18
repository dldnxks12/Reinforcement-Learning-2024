import sys
import string
import numpy as np

rewards = []
com = 'InvertedPendulum-v4_obs_'
com2 = '_max_obs_'
com3 = '_seed_(1)_trial_'
obs     = 15
max_obs = 15
trial   = 4

with open(com + str(obs) + com2 + str(max_obs) + com3 + str(trial) + '.txt', 'r') as file:
    data = file.readlines()
    for d in data:
        d = d.split()
        rewards.append(d[1])
rewards = np.array(rewards)
np.save('./np-files-F/'+ com + str(obs) + com2 + str(max_obs) + com3 + str(trial) +'.npy', rewards)
