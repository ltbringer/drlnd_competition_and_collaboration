import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.initialize_weights import hidden_init
from constants import constants


class Critic(nn.Module):
    """
    Critic: Predicts the value for the policy chosen by the actor.
    """
    def __init__(
            self,
            state_size,
            action_size,
            seed=0,
            fcs1_units=constants.FC1_UNITS,
            fc2_units=constants.FC2_UNITS
    ):
        """
        :param state_size: Dimension of each state
        :param action_size: Dimension of each action
        :param seed: Random seed
        :param fcs1_units: Number of nodes in the first hidden layer
        :param fc2_units: Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed   = torch.manual_seed(seed)
        self.fcs1   = nn.Linear(state_size, fcs1_units)
        self.bn1    = nn.BatchNorm1d(fcs1_units)
        self.d1     = nn.Dropout(p=0.1)
        self.fc2    = nn.Linear(fcs1_units+action_size, fc2_units)
        self.d2     = nn.Dropout(p=0.1)
        self.fc3    = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """
        Build a critic network that maps (state, action) pairs -> Q-values.
        :param state:
        :param action:
        :return:
        """
        # since nn.BatchNorm1d(fc1_units) Applies Batch Normalization over a 2D or 3D input
        # source: https://pytorch.org/docs/stable/nn.html#batchnorm1d
        if state.dim() == 1:
            state = torch.unsqueeze(state,0)

        xs = self.d1(self.bn1(F.relu(self.fcs1(state))))
        x = torch.cat((xs, action), dim=1)
        x = self.d2(F.relu(self.fc2(x)))
        return self.fc3(x)
