import argparse
import torch
import random
from sac import SAC
from trainer import Trainer
from utils import set_seed, make_delayed_env, make_env

if __name__ == '__main__':
    for trial in range(0, 5):
        parser = argparse.ArgumentParser()

        parser.add_argument('--env-name', default='Walker2d-v3', type=str)

        # TODO : Select augmented state dimension & O_max
        min_obs_delayed = 0
        max_obs_delayed = 5
        init_obs_delay  = random.randrange(1, max_obs_delayed+1)

        parser.add_argument('--init-obs-delayed-steps',  default=init_obs_delay,  type=int)
        parser.add_argument('--min-obs-delayed-steps',   default=min_obs_delayed, type=int)
        parser.add_argument('--max-obs-delayed-steps',   default=max_obs_delayed, type=int)

        parser.add_argument('--random-seed', default=1, type=int)
        parser.add_argument('--eval_flag', default=True, type=bool)
        parser.add_argument('--eval-freq', default=5000, type=int)
        parser.add_argument('--eval-episode', default=5, type=int)
        parser.add_argument('--automating-temperature', default=True, type=bool)
        parser.add_argument('--temperature', default=0.2, type=float)
        parser.add_argument('--start-step', default=10000, type=int)
        parser.add_argument('--max-step', default=2000000, type=int)
        parser.add_argument('--update_after', default=1000, type=int)
        parser.add_argument('--hidden-dims', default=(256, 256))
        parser.add_argument('--batch-size', default=256, type=int)
        parser.add_argument('--buffer-size', default=1000000, type=int)
        parser.add_argument('--update-every', default=50, type=int)
        parser.add_argument('--log_std_bound', default=[-20, 2])
        parser.add_argument('--gamma', default=0.99, type=float)
        parser.add_argument('--actor-lr', default=3e-4, type=float)
        parser.add_argument('--critic-lr', default=3e-4, type=float)
        parser.add_argument('--temperature-lr', default=3e-4, type=float)
        parser.add_argument('--tau', default=0.005, type=float)
        parser.add_argument('--show-loss', default=False, type=bool)
        args = parser.parse_args()

        # Set Device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Set Seed
        trial_seed = args.random_seed + trial
        random_seed = set_seed(trial_seed)

        # Create Delayed Environment
        env , eval_env = make_delayed_env(args, random_seed)

        state_dim     = env.observation_space.shape[0]
        action_dim    = env.action_space.shape[0]
        action_bound  = [env.action_space.low[0], env.action_space.high[0]]

        print("-----------------------------------------------------------------------------------", sep = "")
        print(f"Environment: {args.env_name}, Random Seed: {trial_seed}", sep = "")
        print(f"Init Obs. Delayed: {args.init_obs_delayed_steps}, Min-Obs. Delayed: {args.min_obs_delayed_steps}, Max-Obs. Delayed: {args.max_obs_delayed_steps}", sep = "")
        print("-----------------------------------------------------------------------------------", "\n")

        # Create Agent
        agent = SAC(args, state_dim, action_dim, action_bound, env.action_space, device)

        # Create Trainer & Train
        trainer = Trainer(env, eval_env, agent, trial, args)
        trainer.learning()