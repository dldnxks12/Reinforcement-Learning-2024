import gym
import sys
import random
import numpy as np

"""

SARSA : On-policy TD

"""

env = gym.make('FrozenLake-v1')
env = gym.wrappers.TimeLimit(env, max_episode_steps = 50)

def soft_greedy_policy(Q, state):
    e = 0.3
    if random.random() < e:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :]) # return index
    return action

def td_control(env, n_episode):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    alpha = 0.1
    gamma = 0.9

    for _ in range(n_episode):
        state = env.reset()[0]
        terminated, truncated = False, False

        while (terminated == False) or (truncated == False):

            # TODO : a(t), a(t+1) 모두 epsilon greedy policy에서 선택
            action      = soft_greedy_policy(Q, state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = soft_greedy_policy(Q, next_state)

            if terminated or truncated:
                Q[state][action] += alpha * (reward - Q[state][action])
            else:
                Q[state][action] += alpha * (reward + gamma*Q[next_state][next_action] - Q[state][action])

            state = next_state
    return Q

Q = td_control(env,100000)
print(Q)


def PolicyImprovement(value_table):
    policy = np.argmax(value_table, axis = 1)
    return policy

optimal_policy = PolicyImprovement(Q)
print(optimal_policy)