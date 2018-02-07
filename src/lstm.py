"""
Project:    stock_prediction
File:       lstm.py
Created by: louise
On:         29/01/18
At:         4:56 PM
"""
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np

from src.primaldual.linear_operators import GeneralLinearOperator, GeneralLinearAdjointOperator
from src.primaldual.primal_dual_updates import DualGeneralUpdate, PrimalGeneralUpdate, PrimalRegularization
from src.primaldual.proximal_operators import ProximalLinfBall, ProximalQuadraticForm


class LSTM(nn.Module):
    def __init__(self, hidden_size=64, hidden_size2=128, num_securities=5, dropout=0.2, n_layers=8, T=10, training=True):
        """

        :param hidden_size: int
        :param num_securities: int
        :param dropout: float
        :param training: bool
        """
        super(LSTM, self).__init__()
        self.training = training
        self.hidden_size = hidden_size
        self.hidden_size2 = hidden_size2
        self.rnn = nn.LSTM(
            input_size=num_securities,
            hidden_size=self.hidden_size,
            num_layers=n_layers,
            dropout=dropout,
            bidirectional=False
        )
        # self.rnn2 = nn.LSTM(
        #     input_size=num_securities,
        #     hidden_size=self.hidden_size2,
        #     num_layers=n_layers,
        #     dropout=dropout,
        #     bidirectional=False
        # )

        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc1 = nn.Linear(self.hidden_size, 10)
        #self.bn1 = nn.BatchNorm1d(self.hidden_size)
        self.fc2 = nn.Linear(10, num_securities)
        self.relu = nn.ReLU()
        self.T = T

    def forward(self, x):
        """

        :param x: Pytorch Variable, T x batch_size x n_stocks
        :return:
        """
        batch_size = x.size()[1]
        seq_length = x.size()[0]

        x = x.view(seq_length, batch_size, -1)

        # We need to pass the initial cell states
        h0 = Variable(torch.zeros(self.rnn.num_layers, batch_size, self.hidden_size)).cuda()
        c0 = Variable(torch.zeros(self.rnn.num_layers, batch_size, self.hidden_size)).cuda()
        outputs, (ht, ct) = self.rnn(x, (h0, c0))
        # h1 = Variable(torch.zeros(self.rnn.num_layers, batch_size, self.hidden_size2)).cuda()
        # c1 = Variable(torch.zeros(self.rnn.num_layers, batch_size, self.hidden_size2)).cuda()
        # outputs, (ht, ct) = self.rnn2(x, (h1, c1))
        out = outputs[-1]  # We are only interested in the final prediction
        out = self.fc1(out)
        #out = self.bn1(out)
        out = self.relu(out)
        out = F.dropout(out, training=self.training)
        out = self.fc2(out)
        return out


def predict_sequences_multiple(model, data, window_size, prediction_len):
    prediction_seqs = []
    for i in range(int(len(data)/prediction_len)):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.forward(curr_frame[np.newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs


class PD_LSTM(nn.Module):
    def __init__(self, hidden_size=64, num_securities=5, dropout=0.2, T=10,
                 max_it=20, sigma=0.5, tau=0.1, theta=0.9,
                 training=True):
        """

        :param hidden_size: int
        :param num_securities: int
        :param dropout: float
        :param training: bool
        """
        super(PD_LSTM, self).__init__()
        self.training = training
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(
            input_size=num_securities,
            hidden_size=hidden_size,
            num_layers=T-1,
            dropout=dropout,
            bidirectional=False, )
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_op = GeneralLinearOperator()
        self.linear_op_adj = GeneralLinearAdjointOperator()
        self.max_it = max_it
        self.dual_update = DualGeneralUpdate(sigma)
        self.prox_l_inf = ProximalLinfBall()
        self.prox_quad = ProximalQuadraticForm()
        self.primal_update = PrimalGeneralUpdate(lambda_rof, tau)
        self.primal_reg = PrimalRegularization(theta)

        self.pe = 0.0
        self.de = 0.0
        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.w = w
        self.clambda = nn.Parameter(lambda_rof.data)
        self.sigma = nn.Parameter(sigma.data)
        self.tau = nn.Parameter(tau.data)
        self.theta = nn.Parameter(theta.data)

    def forward(self, x):
        """

        :param x: Pytorch Variable, T x batch_size x n_stocks
        :return:
        """
        batch_size = x.size()[1]
        seq_length = x.size()[0]

        x = x.view(seq_length, batch_size, -1)

        # We need to pass the initial cell states
        h0 = Variable(torch.zeros(seq_length, batch_size, self.hidden_size)).cuda()
        c0 = Variable(torch.zeros(seq_length, batch_size, self.hidden_size)).cuda()
        outputs, (ht, ct) = self.rnn(x, (h0, c0))

        out = outputs[-1]  # We are only interested in the final prediction
        out = self.fc1(out)
        out = self.relu(out)
        out = F.dropout(out, training=self.training)
        out = self.fc2(out)
        return out