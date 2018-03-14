"""
Project:    stock_prediction
File:       lstm.py
Created by: louise
On:         08/02/18
At:         12:55 PM
"""
import torch
from torch import nn
from torch.autograd import Variable


class LSTM(nn.Module):
    def __init__(self, hidden_size=64, hidden_size2=128, num_securities=5, dropout=0.2, n_layers=8, T=10):
        """
        Constructor of the LSTM based NN for time series prediction

        :param hidden_size: int, size of the first hidden layer
        :param hidden_size2: int, size of second hidden layer
        :param num_securities: int, number of stocks being predicted at the same time
        :param dropout: float, dropout value
        :param n_layers: int, number of layers
        :param T:
        """
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_size2 = hidden_size2
        self.rnn = nn.LSTM(
            input_size=num_securities,
            hidden_size=self.hidden_size,
            num_layers=n_layers,
            dropout=dropout,
            bidirectional=False
        )

        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc1.weight.data.normal_()
        self.fc3 = nn.Linear(self.hidden_size, 10)
        self.fc2 = nn.Linear(10, num_securities)
        self.relu = nn.ReLU()
        self.T = T

    def forward(self, x):
        """
        forward function for LSTM net.
        :param x: Pytorch Variable, T x batch_size x n_stocks
        :return:
        """
        batch_size = x.size()[1]
        seq_length = x.size()[0]

        x = x.view(seq_length, batch_size, -1)  # just to be sure of the dimensions

        # Initial cell states
        h0 = Variable(torch.zeros(self.rnn.num_layers, batch_size, self.hidden_size)).cuda()
        c0 = Variable(torch.zeros(self.rnn.num_layers, batch_size, self.hidden_size)).cuda()
        outputs, (ht, ct) = self.rnn(x, (h0, c0))
        out = outputs[-1]  # last prediction
        out = self.fc1(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out
