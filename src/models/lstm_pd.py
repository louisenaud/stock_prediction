"""
Project:    stock_prediction
File:       lstm_pd.py
Created by: louise
On:         08/02/18
At:         12:55 PM
"""
import torch
from torch import nn
from torch.autograd import Variable

from src.primaldual.linear_operators import GeneralLinearOperator, GeneralLinearAdjointOperator
from src.primaldual.primal_dual_updates import DualGeneralUpdate, PrimalGeneralUpdate, PrimalRegularization
from src.primaldual.proximal_operators import ProximalLinfBall, ProximalQuadraticForm


class PD_LSTM(nn.Module):
    """
    LSTM network with Primal Dual recurrent network. This model uses LSTM to encode the time series, then the primal
    dual approach to get a structured term in the loss.
    Then two linear layers are applied to get a prediction.
    """
    def __init__(self, H, b, hidden_size=64, n_layers=2, num_securities=5, dropout=0.2,
                 max_it=20, sigma=0.5, tau=0.1, theta=0.9, lambda_rof=5.,
                 training=True):
        """

        :param hidden_size: int
        :param num_securities: int
        :param dropout: float
        :param T: int
        :param max_it: int
        :param sigma: float
        :param tau: float
        :param theta: float
        :param training: float
        """
        super(PD_LSTM, self).__init__()
        self.training = training
        # LSTM
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.rnn = nn.LSTM(
            input_size=num_securities,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout,
            bidirectional=False, )

        # Primal Dual RNN
        self.linear_op = GeneralLinearOperator()
        self.linear_op_adj = GeneralLinearAdjointOperator()
        self.max_it = max_it
        self.prox_l_inf = ProximalLinfBall()
        self.prox_quad = ProximalQuadraticForm()
        self.primal_reg = PrimalRegularization(theta)

        self.pe = 0.0
        self.de = 0.0
        self.clambda = nn.Parameter(lambda_rof * torch.ones(1).type_as(H.data))
        self.sigma = nn.Parameter(sigma * torch.ones(1).type_as(H.data))
        self.tau = nn.Parameter(tau * torch.ones(1).type_as(H.data))
        self.theta = nn.Parameter(theta * torch.ones(1).type_as(H.data))
        self.primal_update = PrimalGeneralUpdate(self.tau)
        self.dual_update = DualGeneralUpdate(self.sigma)

        self.H = H
        self.b = b

        # Linear layers
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc1.weight.data.normal_()
        self.fc2 = nn.Linear(self.hidden_size, num_securities)

    def forward(self, x, x_obs):
        """

        :param x: Pytorch Variable, T x batch_size x n_stocks, current estimated sequence
        :param x_obs: Pytorch Variable, T x batch_size x n_stocks, observed sequence
        :return:
        """
        batch_size = x.size()[1]
        seq_length = x.size()[0]
        n_stocks = x.size()[2]

        x = x.view(seq_length, batch_size, -1)

        # Encode time series through LSTM cells
        h0 = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)).cuda()
        c0 = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)).cuda()
        outputs, (ht, ct) = self.rnn(x, (h0, c0))  # seq_length x batch_size x n_stocks

        # Initialize variables for Primal Dual Net
        x_tilde = Variable(outputs.data.permute(1, 0, 2).clone()).type_as(x)  # batch_size x seq_length x hidden
        x = Variable(outputs.data.permute(1, 0, 2).clone()).type_as(x)  # batch_size x seq_length x hidden
        y = Variable(torch.ones((2, seq_length, self.hidden_size))).type_as(x)  # batch_size x 2 x seq_length x hidden
        x_y = Variable(torch.ones((2, seq_length, self.hidden_size))).type_as(x)  # batch_size x 2 x seq_length x hidden
        # Forward pass

        self.theta.data.clamp_(0, 5)
        for it in range(self.max_it):
            # Dual update
            Lx = self.linear_op.forward(x_tilde)  # Bx2xTxn_stocks
            y = self.dual_update.forward(x_tilde.unsqueeze(1), Lx)  # Bx2xTxn_stocks
            y = self.prox_l_inf.forward(y, 1.0)  # Bx2xTxn_stocks
            # Primal update
            x_old = x
            Ladjy = self.linear_op_adj.forward(y)  # Bx1xTxn_stocks
            x = self.primal_update.forward(outputs, Ladjy)  # Bx1xTxn_stocks
            x = self.prox_quad.forward(x, self.H, self.b, self.tau)  # 1xTxn_stocks
            # Smoothing
            x = x.view(-1, seq_length, self.hidden_size)
            x_tilde = self.primal_reg.forward(x, x_tilde, x_old)  # Bx1xTxn_stocks

        out = self.fc1(x_tilde)
        out = self.fc2(out)
        return out[:, -1, :]
