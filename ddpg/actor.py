import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.initialize_weights import hidden_init
from constants import constants


class Actor(nn.Module):
    """
    Actor: Attempts to predict the optimal Policy Q*
    """
    def __init__(
            self,
            state_size,
            action_size,
            seed=0,
            fc1_units=constants.FC1_UNITS,
            fc2_units=constants.FC2_UNITS
    ):
        """
        :param state_size: Dimension of each state
        :param action_size: Dimension of each action
        :param seed: Random seed
        :param fc1_units: Number of nodes in first hidden layer
        :param fc2_units: Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed   = torch.manual_seed(seed)
        self.fc1    = nn.Linear(state_size, fc1_units)
        self.bn1    = nn.BatchNorm1d(fc1_units)
        self.d1     = nn.Dropout(p=0.1)
        self.fc2    = nn.Linear(fc1_units, fc2_units)
        self.d2     = nn.Dropout(p=0.1)
        self.fc3    = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """
        Build an actor (policy) network that maps states -> actions.
        :param state:
        :return:
        """

        # since nn.BatchNorm1d(fc1_units) Applies Batch Normalization over a 2D or 3D input
        # source: https://pytorch.org/docs/stable/nn.html#batchnorm1d
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)
        x = self.d1(self.bn1(F.relu(self.fc1(state))))
        x = self.d2(F.relu(self.fc2(x)))
        return F.tanh(self.fc3(x))
