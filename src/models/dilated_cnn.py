"""
Project:    stock_prediction
File:       dilated_cnn.py
Created by: louise
On:         20/02/18
At:         1:42 PM
"""
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np


class DilatedNet(nn.Module):
    def __init__(self, num_securities=5, n_layers=8, T=10, training=True):
        """

        :param hidden_size: int
        :param num_securities: int
        :param dropout: float
        :param training: bool
        """
        super(DilatedNet, self).__init__()
        self.training = training
        self.dilation = 2
        # First Layer
        # Input
        self.dilated_conv1 = nn.Conv1d(1, 1, kernel_size=num_securities, dilation=2)
        self.relu1 = nn.ReLU()
        # conditionnal branch
        self.dilated_conv_cond = nn.Conv1d(num_securities-1, 1, kernel_size=num_securities-1, dilation=2)
        self.relu_cond = nn.ReLU()

        # Layer 2
        self.dilated_conv2 = nn.Conv1d(1, 1, dilation=2)
        self.relu2 = nn.ReLU()

        # Output layer
        self.conv_final = nn.Conv1d(1, 1)

        self.T = T

    def forward(self, x):
        """

        :param x: Pytorch Variable, T x batch_size x n_stocks
        :return:
        """
        batch_size = x.size()[1]
        seq_length = x.size()[0]
        # Make sure x has the right dimensions
        x = x.view(seq_length, batch_size, -1)
        xx = x[:, :, 0]
        xy = x[:, :, 1:]
        # First layer
        out = self.dilated_conv1(xx)
        out = self.relu1(out)
        out += xx

        # Conditional branch
        out_c = self.dilated_conv_cond(xy)
        out_c = self.relu_cond(out_c)
        out_c += xy
        # conditionning
        out = out + out_c

        # Layer 2:
        out = self.dilated_conv2(out)
        out = self.relu2(out)

        # Final layer
        out = self.conv_final(out)

        return out
