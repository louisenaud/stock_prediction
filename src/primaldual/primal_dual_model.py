"""
Project:    stock_prediction
File:       primal_dual_model.py
Created by: louise
On:         02/02/18
At:         5:03 PM
"""
from torch.autograd import Variable
import torch
import torch.nn as nn


from primal_dual_updates import PrimalWeightedUpdate, PrimalRegularization, DualWeightedUpdate
from proximal_operators import ProximalLinfBall
from linear_operators import GeneralLinearOperator, GeneralLinearAdjointOperator


class GeneralNet(nn.Module):

    def __init__(self, w1, w2, w, max_it, lambda_rof, sigma, tau, theta, dtype=torch.cuda.FloatTensor):
        """
        Constructor of the Primal Dual Net.
        :param w1: Pytorch variable [2xMxN]
        :param w2: Pytorch variable [2xMxN]
        :param w: Pytorch variable [2xMxN]
        :param max_it: int
        :param lambda_rof: float
        :param sigma: float
        :param tau: float
        :param theta: float
        :param dtype: Pytorch Tensor type, torch.cuda.FloatTensor by default.
        """
        super(GeneralNet, self).__init__()
        self.linear_op = GeneralLinearOperator()
        self.linear_op_adj = GeneralLinearAdjointOperator()
        self.max_it = max_it
        self.dual_update = DualWeightedUpdate(sigma)
        self.prox_l_inf = ProximalLinfBall()
        self.primal_update = PrimalWeightedUpdate(lambda_rof, tau)
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

        self.type = dtype

    def forward(self, x, img_obs):
        """
        Forward function for the Net model.
        :param x: Pytorch variable [1xMxN]
        :param img_obs: Pytorch variable [1xMxN]
        :return: Pytorch variable [1xMxN]
        """
        x = Variable(img_obs.data.clone()).cuda()
        x_tilde = Variable(img_obs.data.clone()).cuda()
        img_size = img_obs.size()
        y = Variable(torch.ones((img_size[0] + 1, img_size[1], img_size[2]))).cuda()

        # Forward pass
        y = self.linear_op(x)
        w_term = Variable(torch.exp(-torch.abs(y.data.expand_as(y))))
        self.w = self.w1.expand_as(y) + self.w2.expand_as(y) * w_term
        self.w.type(self.type)
        self.theta.data.clamp_(0, 5)
        for it in range(self.max_it):
            # Dual update
            y = self.dual_update.forward(x_tilde, y, self.w)
            y.data.clamp_(0, 1)
            y = self.prox_l_inf.forward(y, 1.0)
            # Primal update
            x_old = x
            x = self.primal_update.forward(x, y, img_obs, self.w)
            x.data.clamp_(0, 1)
            # Smoothing
            x_tilde = self.primal_reg.forward(x, x_tilde, x_old)
            x_tilde.data.clamp_(0, 1)

        return x