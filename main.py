import random, argparse, logging, os
import numpy as np
import gym
import torch
from torch.utils.tensorboard import SummaryWriter

from q_learning import init_dqn, q_learning_loss
from arg_setting import parse_arguments
from utils import test_policy, save_policy

def train_dqn(args):
    """Runs DQN training procedure.
    """
    # setup TensorBoard logging
    writer = SummaryWriter(log_dir=args.logdir)
    # make environment
    env = gym.make(args.env_ID)
    env.seed(args.random_seed)
    # instantiate reward model + buffers and optimizers for training DQN
    q_net, q_target, replay_buffer, optimizer_agent, epsilon_schedule = init_dqn(args)
    # begin training
    ep_return = 0
    i_episode = 0
    state = env.reset()
    for step in range(args.n_agent_steps):
        # agent interact with env
        epsilon = epsilon_schedule.value(step)
        action = q_net.act(state, epsilon)
        next_state, rew, done, _ = env.step(action)
        # record step info
        replay_buffer.push(state, action, rew, next_state, done)
        ep_return += rew
        # prepare for next step
        state = next_state
        if done:
            state = env.reset()
            writer.add_scalar('1.ep_return', ep_return, step)
            ep_return = 0
            i_episode += 1

            # q_net gradient step at end of each episode
            if step >= args.agent_learning_starts and len(replay_buffer) >= 3*args.batch_size_agent: # we now make learning updates at the end of every episode
                loss = q_learning_loss(q_net, q_target, replay_buffer, args)
                optimizer_agent.zero_grad()
                loss.backward()
                optimizer_agent.step()
                writer.add_scalar('3.loss', loss, step)
                writer.add_scalar('4.epsilon', epsilon, step)
            if args.epsilon_annealing_scheme == 'exp':
                epsilon_schedule.step()

        # update q_target
        if step % args.target_update_period == 0: # update target parameters
            for target_param, local_param in zip(q_target.parameters(), q_net.parameters()):
                target_param.data.copy_(q_net.tau*local_param.data + (1.0-q_net.tau)*target_param.data)
        
        # evalulate agent performance
        if step > 0 and step % args.agent_test_period == 0 or step == args.n_agent_steps - 1:
            logging.info("Agent has taken {} steps. Testing performance for 100 episodes".format(step))
            mean_ep_return = test_policy(q_net, args, writer)
            writer.add_scalar('2.mean_ep_return_test', mean_ep_return, step)
            # save current policy
            save_policy(q_net, optimizer_agent, step, args)
            # Possibly end training if mean_ep_return is above the threshold
            if env.spec.reward_threshold != None and mean_ep_return >= env.spec.reward_threshold:
                raise SystemExit("Environment solved after {} episodes!".format(i_episode))
    writer.close()


def main():
    """Sets up experiment.
    """
    args = parse_arguments() # parse command line arguments (experiment settings)
    # setup logging
    os.makedirs('./logs/', exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    # logging.getLogger().addHandler(logging.StreamHandler()) # makes messages print to stderr, too
    logging.info('Running experiment with the following settings:')
    for arg in vars(args):
        logging.info('{}: {}'.format(arg, getattr(args, arg)))
    args.logdir = './logs/{}/{}'.format(args.env_ID, args.random_seed)
    os.makedirs('{}/checkpts/'.format(args.logdir), exist_ok=True) # dir for saving model at checkpoints

    # for reproducibility
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    # run experiment!
    train_dqn(args)
    
if __name__ == '__main__':
    main()