"""
Project:    pytorch_primal_dual
File:       proximal_operators.py
Created by: louise
On:         29/11/17
At:         3:54 PM
"""
from torch.autograd import Variable
import torch
import torch.nn as nn


class ProximalLinfBall(nn.Module):
    def __init__(self):
        super(ProximalLinfBall, self).__init__()

    def forward(self, p, r, dtype=torch.cuda.FloatTensor):
        """
        Computes the proximal operator of Linf, with parameter r.
        :param p: PyTorch Variable
        :param r: float
        :param dtype: tensor type
        :return: PyTorch Variable
        """

        m1 = torch.max(torch.add(p.data, - r).type(dtype), torch.zeros(p.size()).type(dtype))
        m2 = torch.max(torch.add(torch.neg(p.data), - r).type(dtype), torch.zeros(p.size()).type(dtype))

        return p - Variable(m1 - m2)


class ProximalL1(nn.Module):
    def __init__(self):
        super(ProximalL1, self).__init__()

    def forward(self, x, f, clambda):
        """
        Computes the proximal operator of L1, with parameter r.
        :param x: PyTorch Variable, [1xMxN]
        :param f: PyTorch Variable, [1xMxN]
        :param clambda: float
        :return: PyTorch Variable [1xMxN]
        """
        if x.is_cuda:
            res = x + torch.clamp(f - x, -clambda, clambda).cuda()
        else:
            res = x + torch.clamp(f - x, -clambda, clambda)
        return res


class ProximalQuadraticForm(nn.Module):
    def __init__(self):
        super(ProximalQuadraticForm, self).__init__()

    def forward(self, x, H, b, tau):
        """
        Computes the proximal operator for a quadratic form 1/2 x^T H + b^T x (+ c)
        :param x: PyTorch Variable, [1xM*N]
        :param H: PyTorch Variable, [M*NxM*N]
        :param b: PyTorch Variable, [1xM*N]
        :param tau: PyTorch Variable, [1]
        :return: PyTorch Variable, [1xM*N]
        """
        P = tau.expand_as(H) * H + Variable(torch.eye(H.size()[0])).type_as(H)
        P_inv = torch.inverse(P)
        X = x.view(-1, 1) - tau.expand_as(b) * b
        return P_inv.matmul(X)
