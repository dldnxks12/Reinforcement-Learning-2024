import sys
import numpy as np
import gymnasium as gym

env = gym.make('FrozenLake-v1', map_name = '8x8', is_slippery=False)

def policy_evaluation(policy, env, discount_factor = 1.0, theta = 0.00001):
    V = np.zeros(env.observation_space.n)

    while True:
        delta = 0

        for s in range(env.observation_space.n):
            v = 0
            for action, action_prob in enumerate(policy[s]): # 가능한 action들로..
                for transition_prob, next_state, reward, done in env.P[s][action]: # next_state로..
                    v += action_prob * transition_prob * (reward + V[next_state])
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break

    return np.array(V)


random_policy = np.ones([env.observation_space.n, env.action_space.n]) / env.action_space.n

v = policy_evaluation(random_policy, env)

print("Value function :")
print(v)

