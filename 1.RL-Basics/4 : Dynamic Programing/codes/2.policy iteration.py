import sys
import numpy as np
import gymnasium as gym

def PolicyIteration(env, discount_factor):

    # TODO : randomly initialized policy
    random_policy = np.ones(env.action_space.n) / env.action_space.n # 1/4
    iterations = 200000

    for i in range(iterations):

        # TODO : Do GPI
        new_value_function = PolicyEvaluation(env, random_policy, discount_factor)
        new_policy         = PolicyImprovement(new_value_function)

        # TODO : No improvement >> Stop
        if(np.all(random_policy == new_policy)):
            break

        random_policy = new_policy

    return new_policy, new_value_function

def PolicyEvaluation(env, policy, discount_factor, theta = 1e-6):
    Q = np.zeros([env.observation_space.n, env.action_space.n])

    while True:
        delta = 0
        for s in range(env.observation_space.n):
            q = np.zeros(env.action_space.n)
            for a in range(env.action_space.n):
                for transition_prob, next_state, reward, done in env.P[s][a]:
                    q[a] += transition_prob * (reward + discount_factor*np.max(Q[next_state]))
                delta = max(delta, np.abs(q[a] - Q[s][a]))
            for i in range(env.action_space.n):
                Q[s][i] = q[i]

        if delta < theta:
            break
    return Q

def PolicyImprovement(value_table):
    policy = np.argmax(value_table, axis = 1)
    return policy


env = gym.make('FrozenLake-v1', map_name = '4x4', is_slippery=False)

optimal_policy, optimal_value_function = PolicyIteration(env, discount_factor = 0.9)

print(optimal_value_function)
print(optimal_policy)









