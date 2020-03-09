"""Generic utilities for testing and saving deep RL agents"""

import torch, gym

def test_policy(q_net, args, writer, num_episodes=100):
    """Using the non-continuous version of the environment and q_net
       with argmax policy (deterministic), run the polcy for
       `num_episodes` and log episode returns.
       Returns mean return over `num_episodes` episodes.
    """
    env = gym.make(args.env_ID) # make a fresh env for testing in
    env.seed(args.random_seed)
    i_episode = 0
    ep_return = 0
    total_return = 0
    state = env.reset()
    while i_episode < num_episodes:
        # agent interact with env
        action = q_net.act(state, epsilon=0)
        assert env.action_space.contains(action)
        next_state, r_true, done, _ = env.step(action)
        ep_return += r_true
        # prepare for next step
        state = next_state
        if done:
            total_return += ep_return
            ep_return = 0
            i_episode += 1
            state = env.reset()
    return total_return / num_episodes


def save_policy(q_net, policy_optimizer, step, args):
    path = '{}/checkpts/step_{}.pt'.format(args.logdir, step)
    torch.save({
        'policy_state_dict': q_net.state_dict(),
        'policy_optimizer_state_dict': policy_optimizer.state_dict(),
        }, path)