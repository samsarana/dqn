"""Q-network architectures"""

import random, torch
import torch.nn as nn

class DQN(nn.Module):
    """Deep Q-network parameterised by MLP with two hidden layers
    """
    def __init__(self, num_inputs, num_actions, args):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, args.h1_agent),
            nn.ReLU(),
            nn.Linear(args.h1_agent, args.h2_agent),
            nn.ReLU(),
            nn.Linear(args.h2_agent, num_actions)
        )
        self.num_actions = num_actions
        self.batch_size = args.batch_size_agent
        self.gamma = args.gamma
        self.tau = args.target_update_tau

    def forward(self, x):
        return self.layers(x)
    
    def act(self, state, epsilon=0):
        """For Q-networks, computing action and forward pass are NOT
           the same thing! (unlike for policy networks)
           Takes Box(4) and returns Discrete(2)
           Box(4) = R^4, represented as np.ndarray, shape=(4,), dtype=float32
           Discrete(2) = {0, 1} where 0 and 1 are standard integers
        """
        if random.random() > epsilon:
            state = torch.tensor(state, dtype=torch.float)
            q_value = self(state)
        # max is along non-batch dimension, which may be 0 or 1 depending on whether input is batched (action selection: not batched; computing loss: batched)
            _, action_tensor  = q_value.max(-1) # max returns a (named)tuple (values, indices) where values is the maximum value of each row of the input tensor in the given dimension dim. And indices is the index location of each maximum value found (argmax).
            action = int(action_tensor)
        else:
            action = random.randrange(0, self.num_actions)
        return action


class CnnDQN(nn.Module):
    """Deep Q-network parameterised by CNN
    """
    def __init__(self, num_inputs, num_actions, args):
        super().__init__()
        self.num_actions = num_actions
        self.batch_size = args.batch_size_agent
        self.gamma = args.gamma
        self.tau = args.target_update_tau
        self.convolutions = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4), # num_inputs == env.observation_space.shape[0] == (84,84,4)[0]. Still not sure this is going to work -- can other 2 input dims be left implicitly wtih Conv2d layers? Maybe need to use Conv3d..?
            nn.ReLU(), # TODO should we use dropout and/or batchnorm in between conv layers, as in reward model?
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 512, kernel_size=7, stride=1),
            nn.ReLU(), # TODO should there be a ReLU here?
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = x.view(-1, 3, 84, 84)
        x = self.convolutions(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x
    
    def act(self, state, epsilon=0):
        """For Q-networks, computing action and forward pass are NOT
            the same thing! (unlike for policy networks)
            Takes Box(4) and returns Discrete(2)
            Box(4) = R^4, represented as np.ndarray, shape=(4,), dtype=float32
            Discrete(2) = {0, 1} where 0 and 1 are standard integers
            [NB previously in my Gridworld: took a tuple (int, int) and returns scalar tensor with dtype torch.long]
        """
        if random.random() > epsilon:
            state = torch.tensor(state, dtype=torch.float)
            q_value = self(state)
        # max is along non-batch dimension, which may be 0 or 1 depending on whether input is batched (action selection: not batched; computing loss: batched)
            _, action_tensor  = q_value.max(-1) # max returns a (named)tuple (values, indices) where values is the maximum value of each row of the input tensor in the given dimension dim. And indices is the index location of each maximum value found (argmax).
            action = int(action_tensor)
        else:
            action = random.randrange(0, self.num_actions)
        return action