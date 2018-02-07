"""
Project:    pytorch_primal_dual
File:       primal_dual_updates.py
Created by: louise
On:         29/11/17
At:         3:55 PM
"""
from torch.autograd import Variable
import torch
import torch.nn as nn
from linear_operators import ForwardGradient, ForwardWeightedGradient, BackwardDivergence, BackwardWeightedDivergence, \
    GeneralLinearOperator, GeneralLinearAdjointOperator


class PrimalUpdate(nn.Module):
    def __init__(self, lambda_rof, tau):
        super(PrimalUpdate, self).__init__()
        self.backward_div = BackwardDivergence()
        self.tau = tau
        self.lambda_rof = lambda_rof

    def forward(self, x, y, img_obs, dtype=torch.cuda.FloatTensor):
        """
        Computes classic Chambolle-Pock primal update.
        :param x: PyTorch Variable, [1xMxN]
        :param y: PyTorch Variable, [2xMxN]
        :param img_obs: PyTorch Variable [1xMxN]
        :param dtype: Tensor type
        :return: PyTorch Variable, [1xMxN]
        """
        x = (x + self.tau * self.backward_div.forward(y, dtype) +
             self.lambda_rof * self.tau * img_obs) / (1.0 + self.lambda_rof * self.tau)
        return x


class PrimalWeightedUpdate(nn.Module):
    def __init__(self, lambda_rof, tau):
        super(PrimalWeightedUpdate, self).__init__()
        self.backward_div = BackwardWeightedDivergence()
        self.tau = tau
        self.lambda_rof = lambda_rof

    def forward(self, x, y, img_obs, w, dtype=torch.cuda.FloatTensor):
        """
        Computes Chambolle-Pock primal update with the Forward weighted gradient.
        :param x: PyTorch Variable [1xMxN]
        :param y: PyTorch Variable [2xMxN]
        :param img_obs: PyTorch Variable [1xMxN]
        :return:Pytorch Variable, [1xMxN]
        """

        x = (x + self.tau * self.backward_div.forward(y, w) +
             self.lambda_rof * self.tau * img_obs) / (1.0 + self.lambda_rof * self.tau)
        return x


class PrimalRegularization(nn.Module):
    def __init__(self, theta):
        """
        Constructor of PrimalRegularization module.
        :param theta: Pytorch tensor [1]
        """
        super(PrimalRegularization, self).__init__()
        self.theta = theta

    def forward(self, x, x_tilde, x_old):
        """
        Computes the regularization of the primal variable.
        :param x: PyTorch Variable, [1xMxN]
        :param x_tilde: PyTorch Variable, [1xMxN]
        :param x_old: PyTorch Variable, [1xMxN]
        :return: PyTorch Variable, [1xMxN]
        """
        x_tilde = x + self.theta * (x - x_old)
        return x_tilde


class DualUpdate(nn.Module):
    def __init__(self, sigma):
        """
        Constructor of DualUpdate module.
        :param sigma: Pytorch tensor [1]
        """
        super(DualUpdate, self).__init__()
        self.forward_grad = ForwardGradient()
        self.sigma = sigma

    def forward(self, x_tilde, y):
        """

        :param x_tilde: PyTorch Variable, [1xMxN]
        :param y: PyTorch Variable, [2xMxN]
        :return: PyTorch Variable, [2xMxN]
        """

        y = y + self.sigma * self.forward_grad.forward(x_tilde).type_as(x_tilde)
        return y


class DualWeightedUpdate(nn.Module):
    def __init__(self, sigma):
        """
        Constructor of DualWeightedUpdate module.
        :param sigma: Pytorch tensor [1]
        """
        super(DualWeightedUpdate, self).__init__()
        self.forward_grad = ForwardWeightedGradient()
        self.sigma = sigma

    def forward(self, x_tilde, y, w):
        """

        :param x_tilde: PyTorch Variable, [1xMxN]
        :param y: PyTorch Variable, [2xMxN]
        :param w: PyTorch Variable, [2xMxN]
        :return: PyTorch Variable, [2xMxN]
        """
        y = y + self.sigma.expand_as(y) * self.forward_grad.forward(x_tilde, w)
        return y


class PrimalGeneralUpdate(nn.Module):
    def __init__(self, lambda_rof, tau):
        super(PrimalGeneralUpdate, self).__init__()
        self.backward_div = GeneralLinearAdjointOperator()
        self.tau = tau
        self.lambda_rof = lambda_rof

    def forward(self, x, y, sig_obs, dtype=torch.cuda.FloatTensor):
        """
        Computes General quadratic primal update.
        :param x: PyTorch Variable, [1xMxN]
        :param y: PyTorch Variable, [2xMxN]
        :param sig_obs: PyTorch Variable [1xMxN]
        :param dtype: Tensor type
        :return: PyTorch Variable, [1xMxN]
        """
        x = (x + self.tau * self.backward_div.forward(y, dtype) +
             self.lambda_rof * self.tau * sig_obs) / (1.0 + self.lambda_rof * self.tau)
        return x


class DualGeneralUpdate(nn.Module):
    def __init__(self, sigma):
        """
        Constructor of DualUpdate module.
        :param sigma: Pytorch tensor [1]
        """
        super(DualGeneralUpdate, self).__init__()
        self.forward_grad = GeneralLinearOperator()
        self.sigma = sigma

    def forward(self, x_tilde, y):
        """
        Computes General quadratic Dual Update.
        :param x_tilde: PyTorch Variable, [1xMxN]
        :param y: PyTorch Variable, [2xMxN]
        :return: PyTorch Variable, [2xMxN]
        """

        y = y + self.sigma * self.forward_grad.forward(x_tilde).type_as(x_tilde)
        return y

