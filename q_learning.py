"""Functions for initialising a Q-network and computing the appropriate loss"""

import logging
import torch.optim as optim
import torch.nn.functional as F
from models import DQN, CnnDQN
from replay_buffer import ReplayBuffer
from schedules import LinearSchedule, ExpSchedule

def q_learning_loss(q_net, q_target, replay_buffer, args):
    """Defines the Q-Learning loss function.
       Help on interpreting variables:
       Each dimension of the batch pertains to one transition, i.e. one 5-tuple
            (state, action, reward, next_state, done)
       q_values : batch_dim x n_actions
       q_value : batch_dim (x 1 -> squeezed). tensor of Q-values of action taken
                  in each transition (transitions are sampled from replay buffer)
       next_q_values : batch_dim x n_actions (detached)
       next_q_value : as above, except has Q-values of optimal next action after
                       each transition, rather than action taken by agent
       expected_q_value : batch_dim. implements y_i.
    """
    state, action, rew, next_state, done = replay_buffer.sample(q_net.batch_size)

    q_values         = q_net(state)
    next_q_values    = q_target(next_state).detach() # params from target network held fixed when optimizing loss func

    q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze() # see https://colab.research.google.com/drive/1-6aNmf16JcytKw3BJ2UfGq5zkik1QLFm or https://stackoverflow.com/questions/50999977/what-does-the-gather-function-do-in-pytorch-in-layman-terms
    next_q_value, _  = next_q_values.max(-1) # max returns a (named)tuple (values, indices) where values is the maximum value of each row of the input tensor in the given dimension dim. And indices is the index location of each maximum value found (argmax).
    expected_q_value = rew + q_net.gamma * next_q_value * (1 - done)
    
    if args.dqn_loss == 'mse':
        loss = F.mse_loss(q_value, expected_q_value)
    elif args.dqn_loss == 'huber':
        loss = F.smooth_l1_loss(q_value, expected_q_value)
    else:
        raise NotImplementedError(
            "You haven't implemented {} loss function".format(args.dqn_loss))
    return loss


def init_dqn(args):
    """Intitialises and returns the necessary objects for
       Deep Q-learning:
       Q-network, target network, replay buffer and optimizer.
    """
    logging.info("Initialisaling DQN with architecture {} and optimizer {}".format(args.dqn_archi, args.optimizer_agent))
    if args.dqn_archi == 'mlp':
        q_net = DQN(args.obs_shape, args.n_actions, args)
        q_target = DQN(args.obs_shape, args.n_actions, args)
    elif args.dqn_archi == 'cnn':
        q_net = CnnDQN(args.obs_shape, args.n_actions, args)
        q_target = CnnDQN(args.obs_shape, args.n_actions, args)
    if args.optimizer_agent == 'RMSProp':
        optimizer_agent = optim.RMSprop(q_net.parameters(), lr=args.lr_agent, weight_decay=args.lambda_agent)
    else:
        assert args.optimizer_agent == 'Adam'
        optimizer_agent = optim.Adam(q_net.parameters(), lr=args.lr_agent, weight_decay=args.lambda_agent)
    q_target.load_state_dict(q_net.state_dict()) # set params of q_target to be the same
    replay_buffer = ReplayBuffer(args.replay_buffer_size)

    if args.epsilon_annealing_scheme == 'linear':
        epsilon_schedule = LinearSchedule(schedule_timesteps=int(args.exploration_fraction * args.n_agent_steps),
                                      initial_p=args.epsilon_start,
                                      final_p=args.epsilon_stop)
    else:
        assert args.epsilon_annealing_scheme == 'exp'
        epsilon_schedule = ExpSchedule(decay_rate=args.epsilon_decay, final_p=args.epsilon_stop, initial_p=args.epsilon_start)

    return q_net, q_target, replay_buffer, optimizer_agent, epsilon_schedule