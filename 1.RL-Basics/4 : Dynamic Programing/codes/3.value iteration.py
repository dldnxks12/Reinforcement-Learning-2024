import sys
import gymnasium as gym
import numpy as np

def ValueIteration(env, discount_factor = 0.9, theta = 1e-6):
    Q = np.zeros([env.observation_space.n, env.action_space.n])

    while True:
        delta = 0
        for s in range(env.observation_space.n):
            q = np.zeros(env.action_space.n)

            for a in range(env.action_space.n):
                for transition_prob, next_state, reward, done in env.P[s][a]:
                    q[a] += transition_prob * (reward + discount_factor * np.max(Q[next_state]))

                delta = max(delta, np.abs(q[a] - Q[s][a]))

            for i in range(env.action_space.n):
                Q[s][i] = q[i]

        if delta < theta:
            break

    return np.array(Q)

env = gym.make('FrozenLake-v1', map_name = '4x4', is_slippery=False)

Q = ValueIteration(env)
print(f"Q Value function : {Q}")

# Extract Optimal Policy
Optimal_policy = np.argmax(Q, axis = 1)
print(f"Optimal Policy : {Optimal_policy}")