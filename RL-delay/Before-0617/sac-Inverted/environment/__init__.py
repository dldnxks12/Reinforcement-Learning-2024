from gymnasium.envs.registration import register

register(
    id='InvertedSinglePendulum-v4',
    entry_point="environment.inverted_single_pendulum_v4:InvertedSinglePendulumEnv",
    max_episode_steps=1000
)


"""

register(
        id = 'InvertedTriplePendulum-v4',
        entry_point = "environment.inverted_triple_pendulum_v4:InvertedTriplePendulumEnv",
        max_episode_steps = 1000
)

register(
    id='InvertedSinglePendulum-v4',
    entry_point="environment.inverted_single_pendulum_v4:InvertedSinglePendulumEnv",
    max_episode_steps=1000
)


register(
    id='InvertedDoublePendulum-v4',
    entry_point="environment.inverted_double_pendulum_v4:InvertedDoublePendulumEnv",
    max_episode_steps=1000
)


register(
    id='InvertedQuadruplePendulum-v4',
    entry_point="environment.inverted_quadruple_pendulum_v4:InvertedQuadruplePendulumEnv",
    max_episode_steps=1000
)



"""