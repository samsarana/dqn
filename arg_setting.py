"""Functions for parsing command line arguments (experiment settings)"""

import argparse, gym
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser()
    # experiment settings
    parser.add_argument('--env_ID', type=str, default='CartPole-v0', help='Pick an OpenAI Gym environment with a discrete action space')
    parser.add_argument('--random_seed', type=int, default=0)
    # agent hyperparams
    parser.add_argument('--dqn_archi', type=str, default='mlp', help='Is deep Q-network an `mlp` or `cnn`?')
    parser.add_argument('--dqn_loss', type=str, default='mse', help='Use `mse` or `huber` loss function?')
    parser.add_argument('--optimizer_agent', type=str, default='Adam', help='Use `RMSProp` or `Adam` optimizer?')
    parser.add_argument('--h1_agent', type=int, default=32)
    parser.add_argument('--h2_agent', type=int, default=64)
    parser.add_argument('--h3_agent', type=int, default=None)
    parser.add_argument('--batch_size_agent', type=int, default=32)
    parser.add_argument('--lr_agent', type=float, default=1e-3)
    parser.add_argument('--lambda_agent', type=float, default=1e-4, help='coefficient for L2 regularization for agent optimization')
    parser.add_argument('--replay_buffer_size', type=int, default=30000)
    parser.add_argument('--target_update_period', type=int, default=1)
    parser.add_argument('--target_update_tau', type=float, default=8e-2)
    parser.add_argument('--agent_gdt_step_period', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--epsilon_annealing_scheme', type=str, default='linear', help='linear or exp epsilon annealing?')
    parser.add_argument('--epsilon_start', type=float, default=1.0, help='exploration probability for agent at start')
    parser.add_argument('--epsilon_stop', type=float, default=0.01)
    parser.add_argument('--exploration_fraction', type=float, default=0.1, help='If `linear` epsilon annealing scheme, over what fraction of entire training period is epsilon annealed?')
    parser.add_argument('--epsilon_decay', type=float, default=0.999, help='If `exp` epsilon_annealing_scheme, then `epsilon *= epsilon * epsilon_decay` every learning step, until `epsilon_stop`') 
    parser.add_argument('--n_agent_steps', type=int, default=100000, help='No. of steps that agent takes in environment, while training every `agent_gdt_step_period` steps')
    parser.add_argument('--agent_test_frequency', type=int, default=20, help='Over the course of its `n_agent_steps` steps, how many times is agent performance tested? (and the run terminated if `terminate_once_solved == True`')
    parser.add_argument('--agent_learning_starts', type=int, default=0, help='After how many steps does the agent start making learning updates? This replaced the functionality of n_agent_total_steps.')

    args = parser.parse_args()
    args = add_extra_args(args)
    return args


def add_extra_args(args):
    """Computes and saved some additional arguments which will be helpful
       later on.
    """
    # compute period for testing agent
    assert args.n_agent_steps % args.agent_test_frequency == 0,\
        "agent_test_frequency ({}) should be a factor of n_agent_steps ({})".format(
            args.agent_test_frequency, args.n_agent_steps)
    args.agent_test_period = args.n_agent_steps // args.agent_test_frequency

    # make env in order to some relevant settings in `args` for use in instantiating models
    env = gym.make(args.env_ID)
    if isinstance(env.observation_space, gym.spaces.Box):
        args.obs_shape = env.observation_space.shape[0] # env.observation_space is Box(4,) and calling .shape returns (4,) [gym can be ugly]
    else:
        raise RuntimeError("I don't know what observation space {} is!".format(env.observation_space))
    if isinstance(env.action_space, gym.spaces.Discrete):
        args.n_actions = env.action_space.n # [gym doesn't have a nice way to get shape of Discrete space... env.action_space.shape -> () ]
    else:
        raise NotImplementedError('Only discrete actions supported at the moment, for DQN')
    return args