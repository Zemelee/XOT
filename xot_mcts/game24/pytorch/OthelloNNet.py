# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
sys.path.append('..')
from utils import *

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

class OthelloNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_size = game.getBoardSize() # 4
        self.action_size = game.getActionSize() # 36 所有可能的动作数
        self.hidden_size = 128
        self.args = args # lr/dropout/epochs/batch_size/cuda/num_channels

        super(OthelloNNet, self).__init__()
        # (4-->128-->256-->128-->36-->256-->128-->1)
        self.fc1 = nn.Linear(self.board_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size * 2)

        self.fc_p1 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.fc_p2 = nn.Linear(self.hidden_size, self.action_size) # 

        self.fc_v1 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.fc_v2 = nn.Linear(self.hidden_size, 1)


    def forward(self, s):
        s = s.view(-1, self.board_size) # (batch, 4)
        s = F.relu(self.fc1(s)) # (batch, 4) → (batch, 128)
        s = F.relu(self.fc2(s)) # (batch, 128) → (batch, 256)
        # Policy Head
        sp = self.fc_p1(s)       
        sp = self.fc_p2(sp)   
        # Value Head
        sv = self.fc_v1(s)       
        sv = self.fc_v2(sv)
                                                                               
        # 动作概率分布 价值估计
        return F.log_softmax(sp, dim=1), sv
