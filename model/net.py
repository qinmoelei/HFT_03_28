from logging import raiseExceptions
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import math
import sys

max_punish = 1e12


#Q network
class masked_net1(nn.Module):
    def __init__(self, N_STATES, N_ACTIONS, hidden_nodes):
        super(masked_net1, self).__init__()
        self.fc1 = nn.Linear(N_STATES, hidden_nodes)
        self.fc2 = nn.Linear(2 * hidden_nodes, hidden_nodes)
        self.out = nn.Linear(hidden_nodes, N_ACTIONS)
        self.embedding = nn.Embedding(N_ACTIONS, hidden_nodes)
        self.register_buffer("max_punish", torch.tensor(max_punish))

    def forward(
        self,
        state: torch.tensor,
        previous_action: torch.tensor,
        avaliable_action: torch.tensor,
    ):
        state_hidden = F.relu(self.fc1(state))
        previous_action_hidden = self.embedding(previous_action)
        information_hidden = torch.cat([state_hidden, previous_action_hidden],
                                       dim=1)
        information_hidden = self.fc2(information_hidden)
        action = self.out(information_hidden)
        masked_action = action + (avaliable_action - 1) * self.max_punish
        return masked_action


class masked_net1_with_holding_length(nn.Module):
    def __init__(self, N_STATES, N_ACTIONS, hidden_nodes):
        super(masked_net1_with_holding_length, self).__init__()
        self.fc1 = nn.Linear(N_STATES, hidden_nodes)
        self.fc2 = nn.Linear(3 * hidden_nodes, hidden_nodes)
        self.fc3=nn.Linear(1,hidden_nodes)
        self.out = nn.Linear(hidden_nodes, N_ACTIONS)
        self.embedding = nn.Embedding(N_ACTIONS, hidden_nodes)
        self.register_buffer("max_punish", torch.tensor(max_punish))

    def forward(
        self,
        state: torch.tensor,
        previous_action: torch.tensor,
        avaliable_action: torch.tensor,
        holding_length:torch.tensor
    ):
        state_hidden = F.relu(self.fc1(state))
        previous_action_hidden = self.embedding(previous_action)
        holding_length_hidden=self.fc3(holding_length)
        information_hidden = torch.cat([state_hidden, previous_action_hidden,holding_length_hidden],
                                       dim=1)
        information_hidden = self.fc2(information_hidden)
        action = self.out(information_hidden)
        masked_action = action + (avaliable_action - 1) * self.max_punish
        return masked_action


if __name__ == "__main__":
    N_action = 11
    N_hidden = 32
    state = torch.randn(1, 66)
    previous_action = (torch.distributions.Binomial(10, torch.tensor(
        [0.5] * 1)).sample().long())
    # print(previous_action.shape)
    # print(previous_action)
    avaliable_action = torch.bernoulli(torch.Tensor(1, 11).uniform_(0, 1))
    # print(avaliable_action.type())
    holding_length= torch.randint(low=0,high=10,size=(1, 1)).float()
    print(holding_length.shape)
    # print(holding_length.type())
    net = masked_net1_with_holding_length(66, 11, 32)

    print(
        net(state, previous_action, avaliable_action,holding_length).argmax(dim=1,
                                                             keepdim=True))
