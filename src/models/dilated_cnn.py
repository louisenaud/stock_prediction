"""
Project:    stock_prediction
File:       dilated_cnn.py
Created by: louise
On:         20/02/18
At:         1:42 PM
"""
from torch import nn


class DilatedNet(nn.Module):
    def __init__(self, num_securities=5, hidden_size=64, dilation=2, T=10, training=True):
        """

        :param hidden_size: int
        :param num_securities: int
        :param dropout: float
        :param training: bool
        """
        super(DilatedNet, self).__init__()
        self.training = training
        self.dilation = dilation
        self.hidden_size = hidden_size
        # First Layer
        # Input
        self.dilated_conv1 = nn.Conv1d(num_securities, hidden_size, kernel_size=2, dilation=self.dilation)
        self.relu1 = nn.ReLU()

        # Layer 2
        self.dilated_conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1, dilation=self.dilation)
        self.relu2 = nn.ReLU()

        # Layer 3
        self.dilated_conv3 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1, dilation=self.dilation)
        self.relu3 = nn.ReLU()

        # Layer 4
        self.dilated_conv4 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1, dilation=self.dilation)
        self.relu4 = nn.ReLU()

        # Output layer
        self.conv_final = nn.Conv1d(hidden_size, num_securities, kernel_size=1)

        self.T = T

    def forward(self, x):
        """

        :param x: Pytorch Variable, batch_size x T x n_stocks
        :return:
        """

        # First layer
        out = self.dilated_conv1(x)
        out = self.relu1(out)

        # Layer 2:
        out = self.dilated_conv2(out)
        out = self.relu2(out)

        # Layer 3:
        out = self.dilated_conv3(out)
        out = self.relu3(out)

        # Layer 4:
        out = self.dilated_conv4(out)
        out = self.relu4(out)

        # Final layer
        out = self.conv_final(out)
        out = out[:, :, -1]

        return out


class DilatedNet2D(nn.Module):
    def __init__(self, num_securities=5, n_out=3, hidden_size=64, dilation=1, T=10, training=True):
        """

        :param num_securities: int
        :param hidden_size:
        :param k:
        :param dilation:
        :param n_layers:
        :param T:
        :param training:
        """
        super(DilatedNet2D, self).__init__()
        self.n_out = n_out
        self.training = training
        self.dilation = dilation
        self.hidden_size = hidden_size
        # First Layer
        # Input
        self.dilated_conv1 = nn.Conv2d(1, hidden_size, kernel_size=(1, 2), dilation=(1, self.dilation), padding=(1, 2))
        self.relu1 = nn.ReLU()

        # Layer 2
        self.dilated_conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=(1, 2), dilation=(1, self.dilation))
        self.relu2 = nn.ReLU()

        # Layer 3
        self.dilated_conv3 = nn.Conv2d(hidden_size, hidden_size, kernel_size=(1, 2), dilation=(1, self.dilation))
        self.relu3 = nn.ReLU()

        # Layer 4
        self.dilated_conv4 = nn.Conv2d(hidden_size, hidden_size, kernel_size=(1, 2), dilation=(1, self.dilation))
        self.relu4 = nn.ReLU()

        # Output layer
        self.conv_final = nn.Conv2d(hidden_size, 1, kernel_size=(1, 2))

        self.T = T

    def forward(self, x):
        """

        :param x: Pytorch Variable, batch_size x 1 x T x n_stocks
        :return:
        """

        # First layer
        out = self.dilated_conv1(x)
        out = self.relu1(out)

        # Layer 2:
        out = self.dilated_conv2(out)
        out = self.relu2(out)

        # Layer 3:
        out = self.dilated_conv3(out)
        out = self.relu3(out)

        # Layer 4:
        out = self.dilated_conv4(out)
        out = self.relu4(out)

        # Final layer
        out = self.conv_final(out)
        out = out[:, :, :, -1]

        return out


class DilatedNet2DMultistep(nn.Module):
    def __init__(self, num_securities=5, n_in=20, n_out=3, hidden_size=64, dilation=1, T=10, training=True):
        """

        :param num_securities:
        :param n_in: number of time points in the input
        :param n_out: number of time points in the output
        :param hidden_size: int
        :param dilation: int
        :param T: int, length of lookback
        :param training: bool
        """
        super(DilatedNet2DMultistep, self).__init__()
        self.n_out = n_out
        self.n_in = n_in
        self.training = training
        self.dilation = dilation
        self.hidden_size = hidden_size
        # First Layer
        # Input
        self.dilated_conv1 = nn.Conv2d(1, hidden_size, kernel_size=(1, 2), dilation=(1, self.dilation))
        self.relu1 = nn.ReLU()

        # Layer 2
        self.dilated_conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=(1, 2), dilation=(1, self.dilation))
        self.relu2 = nn.ReLU()

        # Layer 3
        self.dilated_conv3 = nn.Conv2d(hidden_size, hidden_size, kernel_size=(1, 2), dilation=(1, self.dilation))
        self.relu3 = nn.ReLU()

        # Layer 4
        self.dilated_conv4 = nn.Conv2d(hidden_size, hidden_size, kernel_size=(1, 2), dilation=(1, self.dilation))
        self.relu4 = nn.ReLU()

        # Output layer
        self.conv_final = nn.Conv2d(hidden_size, 1, kernel_size=(1, 2))

        self.T = T

    def forward(self, x):
        """

        :param x: Pytorch Variable, batch_size x 1 x T x n_stocks
        :return:
        """

        # First layer
        out = self.dilated_conv1(x)
        out = self.relu1(out)

        # Layer 2:
        out = self.dilated_conv2(out)
        out = self.relu2(out)

        # Layer 3:
        out = self.dilated_conv3(out)
        out = self.relu3(out)

        # Layer 4:
        # out = self.dilated_conv4(out)
        # out = self.relu4(out)

        # Final layer
        out = self.conv_final(out)
        out = out[:, :, :, -self.n_out:]

        return out
