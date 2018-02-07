"""
Project:    stock_prediction
File:       linear_model.py
Created by: louise
On:         06/02/18
At:         12:14 PM
"""
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class FinancialTimeSeriesNetwork(nn.Module):

    def __init__(self):
        """
        From Financial Time Series Prediction Using Deep Learning by Ariel Navon, Yosi Keller

        Section 3.2; Figure 2 has an image of the architecture
        Section 5.2; implementation details of each layer
        Uses ReLU layers; in PyTorch terms this is equal to creating
        nn.Linear() layers, and calling .clamp() in the forward pass
        to compute the weights
        NOTE: .clamp() works for both Tensor and Variable
        """
        super(FinancialTimeSeriesNetwork, self).__init__()

        self.input_layer = nn.Linear(60, 500)
        self.m1 = nn.Linear(500, 200)
        self.m2 = nn.Linear(200, 40)
        self.m3 = nn.Linear(40, 20)
        self.output_layer = nn.Linear(20, 2)

    def forward(self, x):
        x = self.input_layer(x).clamp(min=0)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.m1(x).clamp(min=0)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.m2(x).clamp(min=0)
        x = self.m3(x).clamp(min=0)

        y_pred = self.output_layer(x).clamp(min=0)
        return y_pred
