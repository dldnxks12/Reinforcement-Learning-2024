import sys
import numpy as np
import matplotlib.pyplot as plt


def moving_average(data, window_size=7):
    # Pad the data array to handle edges
    padded_data = np.pad(data, (window_size // 2, window_size // 2), mode='edge')

    # Calculate the moving average using convolution
    weights = np.ones(window_size) / window_size
    smoothed_data = np.convolve(padded_data, weights, mode='valid')

    return smoothed_data

# TODO --------------------------------- FLA ------------------------------------------------------------------------------------------- #

# TODO : Ant
y1 = np.load('Ant-v4_obs_15_max_obs_15_seed_(1)_trial_0.npy')
y1 = np.array([float(i) for i in y1])
y2 = np.load('Ant-v4_obs_15_max_obs_15_seed_(1)_trial_1.npy')
y2 = np.array([float(i) for i in y2])
y3 = np.load('Ant-v4_obs_15_max_obs_15_seed_(1)_trial_2.npy')
y3 = np.array([float(i) for i in y3])
y4 = np.load('Ant-v4_obs_15_max_obs_15_seed_(1)_trial_3.npy')
y4 = np.array([float(i) for i in y4])
y5 = np.load('Ant-v4_obs_15_max_obs_15_seed_(1)_trial_4.npy')
y5 = np.array([float(i) for i in y5])

LA_Ant_mean = np.mean([y1, y2, y3, y4, y5], axis = 0)
LA_Ant_std  = np.std([y1, y2, y3, y4, y5], axis = 0)
LA_Ant_mean = moving_average(LA_Ant_mean)
LA_Ant_std  = moving_average(LA_Ant_std)

# TODO : HalfCheetah
y1 = np.load('HalfCheetah-v4_obs_15_max_obs_15_seed_(1)_trial_0.npy')
y1 = np.array([float(i) for i in y1])
y2 = np.load('HalfCheetah-v4_obs_15_max_obs_15_seed_(1)_trial_1.npy')
y2 = np.array([float(i) for i in y2])
y3 = np.load('HalfCheetah-v4_obs_15_max_obs_15_seed_(1)_trial_2.npy')
y3 = np.array([float(i) for i in y3])
y4 = np.load('HalfCheetah-v4_obs_15_max_obs_15_seed_(1)_trial_3.npy')
y4 = np.array([float(i) for i in y4])
y5 = np.load('HalfCheetah-v4_obs_15_max_obs_15_seed_(1)_trial_4.npy')
y5 = np.array([float(i) for i in y5])

LA_HalfCheetah_mean = np.mean([y1, y2, y3, y4, y5], axis = 0)
LA_HalfCheetah_std  = np.std([y1, y2, y3, y4, y5], axis = 0)
LA_HalfCheetah_mean = moving_average(LA_HalfCheetah_mean)
LA_HalfCheetah_std  = moving_average(LA_HalfCheetah_std)

# TODO : Walker2d
y1 = np.load('Walker2d-v4_obs_15_max_obs_15_seed_(1)_trial_0.npy')
y1 = np.array([float(i) for i in y1])
y2 = np.load('Walker2d-v4_obs_15_max_obs_15_seed_(1)_trial_1.npy')
y2 = np.array([float(i) for i in y2])
y3 = np.load('Walker2d-v4_obs_15_max_obs_15_seed_(1)_trial_2.npy')
y3 = np.array([float(i) for i in y3])
y4 = np.load('Walker2d-v4_obs_15_max_obs_15_seed_(1)_trial_3.npy')
y4 = np.array([float(i) for i in y4])
y5 = np.load('Walker2d-v4_obs_15_max_obs_15_seed_(1)_trial_4.npy')
y5 = np.array([float(i) for i in y5])

LA_Walker2d_mean = np.mean([y1, y2, y3, y4, y5], axis = 0)
LA_Walker2d_std  = np.std([y1, y2, y3, y4, y5], axis = 0)
LA_Walker2d_mean = moving_average(LA_Walker2d_mean)
LA_Walker2d_std  = moving_average(LA_Walker2d_std)

# TODO : Hopper
y1 = np.load('Hopper-v4_obs_15_max_obs_15_seed_(1)_trial_0.npy')
y1 = np.array([float(i) for i in y1])
y2 = np.load('Hopper-v4_obs_15_max_obs_15_seed_(1)_trial_1.npy')
y2 = np.array([float(i) for i in y2])
y3 = np.load('Hopper-v4_obs_15_max_obs_15_seed_(1)_trial_2.npy')
y3 = np.array([float(i) for i in y3])
y4 = np.load('Hopper-v4_obs_15_max_obs_15_seed_(1)_trial_3.npy')
y4 = np.array([float(i) for i in y4])
y5 = np.load('Hopper-v4_obs_15_max_obs_15_seed_(1)_trial_4.npy')
y5 = np.array([float(i) for i in y5])

LA_Hopper_mean = np.mean([y1, y2, y3, y4, y5], axis = 0)
LA_Hopper_std  = np.std([y1, y2, y3, y4, y5], axis = 0)
LA_Hopper_mean = moving_average(LA_Hopper_mean)
LA_Hopper_std  = moving_average(LA_Hopper_std)

# TODO : Pendulum
y1 = np.load('InvertedPendulum-v4_obs_15_max_obs_15_seed_(1)_trial_0.npy')
y1 = np.array([float(i) for i in y1])
y2 = np.load('InvertedPendulum-v4_obs_15_max_obs_15_seed_(1)_trial_1.npy')
y2 = np.array([float(i) for i in y2])
y3 = np.load('InvertedPendulum-v4_obs_15_max_obs_15_seed_(1)_trial_2.npy')
y3 = np.array([float(i) for i in y3])
y4 = np.load('InvertedPendulum-v4_obs_15_max_obs_15_seed_(1)_trial_3.npy')
y4 = np.array([float(i) for i in y4])
y5 = np.load('InvertedPendulum-v4_obs_15_max_obs_15_seed_(1)_trial_4.npy')
y5 = np.array([float(i) for i in y5])

LA_InvertedPendulum_mean = np.mean([y1, y2, y3, y4, y5], axis = 0)
LA_InvertedPendulum_std  = np.std([y1, y2, y3, y4, y5], axis = 0)
LA_InvertedPendulum_mean = moving_average(LA_InvertedPendulum_mean)
LA_InvertedPendulum_std  = moving_average(LA_InvertedPendulum_std)

# TODO --------------------------------- Plot ------------------------------------------------------------------------------------------------ #

f, axes = plt.subplots(2, 3)
f.set_size_inches((14, 8))

x_length  = np.arange(0, len(LA_Ant_mean)) * 5000

# TODO : cheetah
axes[0, 0].plot(x_length, LA_Ant_mean, linestyle = '-', color = 'tab:brown', label = 'FLA-BPQL')
axes[0, 0].fill_between(x_length, LA_Ant_mean+LA_Ant_std, LA_Ant_mean-LA_Ant_std, color = 'tab:brown', alpha = 0.1)
axes[0, 1].plot(x_length, LA_HalfCheetah_mean, linestyle = '-', color = 'tab:brown', label = 'FLA-BPQL')
axes[0, 1].fill_between(x_length, LA_HalfCheetah_mean+LA_HalfCheetah_std, LA_HalfCheetah_mean-LA_HalfCheetah_std, color = 'tab:brown', alpha = 0.1)
axes[0, 2].plot(x_length, LA_Hopper_mean, linestyle = '-', color = 'tab:brown', label = 'FLA-BPQL')
axes[0, 2].fill_between(x_length, LA_Hopper_mean+LA_Hopper_std, LA_Hopper_mean-LA_Hopper_std, color = 'tab:brown', alpha = 0.1)
axes[1, 0].plot(x_length, LA_Walker2d_mean, linestyle = '-', color = 'tab:brown', label = 'FLA-BPQL')
axes[1, 0].fill_between(x_length, LA_Walker2d_mean+LA_Walker2d_std, LA_Walker2d_mean-LA_Walker2d_std, color = 'tab:brown', alpha = 0.1)
axes[1, 1].plot(x_length, LA_InvertedPendulum_mean, linestyle = '-', color = 'tab:brown', label = 'FLA-BPQL')
axes[1, 1].fill_between(x_length, LA_InvertedPendulum_mean+LA_InvertedPendulum_std, LA_InvertedPendulum_mean-LA_InvertedPendulum_std, color = 'tab:brown', alpha = 0.1)

axes[0, 0].grid(True)
axes[0, 0].legend()
axes[0, 0].set_ylim([-400, 8000])
axes[0, 0].set_title("Ant-v4", fontsize = 16)
axes[0, 0].set_xlabel("Steps", fontsize = 12)
axes[0, 0].set_ylabel("Average Return", fontsize = 12)

axes[0, 1].grid(True)
axes[0, 1].legend()
axes[0, 1].set_ylim([-400, 8000])
axes[0, 1].set_title("HalfCheetah-v4", fontsize = 16)
axes[0, 1].set_xlabel("Steps", fontsize = 12)
axes[0, 1].set_ylabel("Average Return", fontsize = 12)

axes[0, 2].grid(True)
axes[0, 2].legend()
axes[0, 2].set_ylim([-400, 4000])
axes[0, 2].set_title("Hopper-v4", fontsize = 16)
axes[0, 2].set_xlabel("Steps", fontsize = 12)
axes[0, 2].set_ylabel("Average Return", fontsize = 12)

axes[1, 0].grid(True)
axes[1, 0].legend()
axes[1, 0].set_ylim([-400, 5000])
axes[1, 0].set_title("Walker2dr-v4", fontsize = 16)
axes[1, 0].set_xlabel("Steps", fontsize = 12)
axes[1, 0].set_ylabel("Average Return", fontsize = 12)

axes[1, 1].grid(True)
axes[1, 1].legend()
axes[1, 1].set_ylim([-400, 1500])
axes[1, 1].set_title("InvertedPendulum-v4", fontsize = 16)
axes[1, 1].set_xlabel("Steps", fontsize = 12)
axes[1, 1].set_ylabel("Average Return", fontsize = 12)

plt.tight_layout()
plt.show()
