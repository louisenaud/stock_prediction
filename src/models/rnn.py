"""
Project:    stock_prediction
File:       rnn.py
Created by: louise
On:         06/02/18
At:         12:17 PM
"""


import torch
import torch.nn as nn
from torch.autograd import Variable

torch.manual_seed(1)


class NNModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, cell):
        """

        :param input_dim:
        :param hidden_dim:
        :param output_dim:
        :param n_layers:
        :param cell:
        """
        super(NNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        if cell == "RNN":
            self.cell = nn.RNN(input_size=self.input_dim, hidden_size=self.hidden_dim,
                               num_layers=self.n_layers, dropout=0.0,
                               nonlinearity="tanh", batch_first=True,)
        if cell == "LSTM":
            self.cell = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim,
                                num_layers=self.n_layers, dropout=0.0,
                               )
        if cell == "GRU":
            self.cell = nn.GRU(input_size=self.input_dim, hidden_size=self.hidden_dim,
                                num_layers=self.n_layers, dropout=0.0, )
        print(self.cell)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)


class RNNModel(NNModel):

    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, cell):

        super(RNNModel, self).__init__(input_dim, hidden_dim, output_dim, n_layers, cell)

    # def init_hidden(self, batch_size):
    #
    #     return hidden

    def forward(self, x, batch_size):

        h0 = Variable(torch.zeros(self.n_layers * 1, batch_size, self.hidden_dim))
        rnnOutput, hn = self.cell(x, h0)  # rnnOutput 12,20,50 hn 1,20,50
        hn = hn.view(batch_size, self.hidden_dim)
        fcOutput = self.fc(hn)

        return fcOutput


# LSTM模型
class LSTMModel(NNModel):

    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, cell):
        """

        :param input_dim:
        :param hidden_dim:
        :param output_dim:
        :param n_layers:
        :param cell:
        """
        super(LSTMModel, self).__init__(input_dim, hidden_dim, output_dim, n_layers, cell)

    def forward(self, x, batch_size):
        """

        :param x:
        :param batch_size:
        :return:
        """
        h0 = Variable(torch.zeros(self.n_layers * 1, batch_size, self.hidden_dim))
        c0 = Variable(torch.zeros(self.n_layers * 1, batch_size, self.hidden_dim))
        out, hn = self.cell(x, (h0, c0))  # rnnOutput 12,20,50 hn 1,20,50
        hn = hn[0].view(batch_size, self.hidden_dim)
        out = self.fc(hn)

        return out


class GRUModel(NNModel):

    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, cell):
        """

        :param input_dim:
        :param hidden_dim:
        :param output_dim:
        :param n_layers:
        :param cell:
        """
        super(GRUModel, self).__init__(input_dim, hidden_dim, output_dim, n_layers, cell)

    def forward(self, x, batch_size):
        """

        :param x:
        :param batch_size:
        :return:
        """
        h0 = Variable(torch.zeros(self.n_layers * 1, batch_size, self.hidden_dim))
        rnnOutput, hn = self.cell(x, h0)  # rnnOutput 12,20,50 hn 1,20,50
        hn = hn.view(batch_size, self.hidden_dim)
        fcOutput = self.fc(hn)

        return fcOutput


class ResRNNModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, net_depth):

        super(ResRNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = 1
        self.net_depth = net_depth
        self.i2h = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.h2h = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.h2o = nn.Linear(self.hidden_dim, self.output_dim, bias=True)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim, bias=True)
        self.ht2h = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        #self.tanh = nn.Tanh()

    def forward(self, x, batch_size):

        h0 = Variable(torch.zeros(self.n_layers * 1, batch_size, self.hidden_dim))
        # output = []
        inputLen = x.data.size()[1]
        ht = h0
        for i in range(inputLen):
            hn = self.i2h(x[:, i, :]) + self.h2h(h0)
            if self.net_depth == 0:
                h0 = nn.Tanh()(hn)
            if self.net_depth == 1:
                # res depth = 1
                h0 = nn.Tanh()(hn + h0)
            if self.net_depth >= 2:
                # res depth = N
                if i % self.resDepth == 0 and i != 0:
                    h0 = nn.Tanh()(hn + ht)
                    ht = hn
                else:
                    h0 = nn.Tanh()(hn)

            if self.net_depth == -1:
                if i == 0:
                    hstart = hn
                if i == inputLen-2:
                    h0 = nn.Tanh()(hn+hstart)
                else:
                    if i % 4 == 0 and i != 0:
                        h0 = nn.Tanh()(hn + ht)
                        ht = hn
                    else:
                        h0 = nn.Tanh()(hn)

        hn = hn.view(batch_size, self.hidden_dim)
        fcOutput = self.fc(hn)

        return fcOutput


class AttentionRNNModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, seq_len):
        super(AttentionRNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = 1
        self.i2h = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.h2h = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.h2o = nn.Linear(self.hidden_dim, self.output_dim, bias=True)
        self.fc = nn.Linear(self.hidden_dim*seq_len, self.output_dim, bias=True)
        # self.tanh = nn.Tanh()

    def forward(self, x, batch_size):
        h0 = Variable(torch.zeros(self.n_layers * 1, batch_size, self.hidden_dim))
        hiddenList = []
        inputLen = x.data.size()[1]
        for i in range(inputLen):
            hn = self.i2h(x[:, i, :]) + self.h2h(h0)
            h0 = nn.Tanh()(hn)
            ht = h0.view(batch_size, self.hidden_dim)
            hiddenList.append(ht)
        flanten = torch.cat(hiddenList, dim=1)

        fcOutput = self.fc(flanten)

        return fcOutput


class ANNModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        """

        :param input_dim:
        :param hidden_dim:
        :param output_dim:
        """
        super(ANNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self,x, batch_size):

        output = self.fc1(x)
        output = self.fc2(output)

        return output


class DecompositionNetModel(nn.Module):

    def __init__(self, input_dim, fchidden_dim, rnnhidden_dim, output_dim):
        """

        :param input_dim:
        :param fchidden_dim:
        :param rnnhidden_dim:
        :param output_dim:
        """
        super(DecompositionNetModel, self).__init__()
        self.fchidden_dim = fchidden_dim
        self.rnnhidden_dim = rnnhidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = 1
        self.rnninput_dim = 1

        # dropout层
        self.drop = nn.Dropout(p=0.3)

        # 一维卷积层
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2,  bias=True)
        self.pool = nn.AvgPool1d(kernel_size=5, stride=1, padding=2)
        #self.conv.weight.data.fill_(0.2)
        self.convWeight = self.conv.weight.data
        #print(self.conv.weight.data)

        # 全连接层
        self.fc1 = nn.Linear(self.input_dim, self.fchidden_dim)
        self.fc2 = nn.Linear(self.fchidden_dim, self.input_dim)

        # 循环神经网络层
        self.rnn1 = nn.RNN(input_size=self.rnninput_dim, hidden_size=self.rnnhidden_dim,
                           num_layers=self.n_layers, dropout=0.5,
                           nonlinearity="tanh", batch_first=True, )
        self.rnn2 = nn.RNN(input_size=self.rnninput_dim, hidden_size=self.rnnhidden_dim,
                          num_layers=self.n_layers, dropout=0.5,
                          nonlinearity="tanh", batch_first=True, )
        self.resrnn1 = ResRNNModel(input_dim=1, hidden_dim=self.rnnhidden_dim, output_dim=1, resDepth=4)
        self.resrnn2 = ResRNNModel(input_dim=1, hidden_dim=self.rnnhidden_dim, output_dim=1, resDepth=4 )
        self.gru1 = nn.GRU(input_size=self.rnninput_dim, hidden_size=self.rnnhidden_dim,
                           num_layers=self.n_layers, dropout=0.0,
                           batch_first=True, )
        self.gru2 = nn.GRU(input_size=self.rnninput_dim, hidden_size=self.rnnhidden_dim,
                           num_layers=self.n_layers, dropout=0.0,
                           batch_first=True, )

        # 线性输出层
        self.fc3 = nn.Linear(self.rnnhidden_dim, self.output_dim)
        self.fc4 = nn.Linear(self.rnnhidden_dim, self.output_dim)

    def forward(self, x, batch_size):
        """

        :param x:
        :param batch_size:
        :return:
        """
        x = torch.unsqueeze(x, 1)
        #print(x.size())
        #x = torch.transpose(x, 1, 2)
        # output = self.fc1(x)
        # prime = self.fc2(output)
        prime = self.conv(x)
        #print(prime.size())
        prime = self.pool(prime)
        #print(prime.size())
        residual = x-prime
        # prime = torch.unsqueeze(prime, 2)
        # residual = torch.unsqueeze(residual, 2)
        prime = torch.transpose(prime, 1, 2)
        residual = torch.transpose(residual, 1, 2)

        h0 = Variable(torch.zeros(self.n_layers * 1, batch_size, self.rnnhidden_dim))

        # 预测主成分rnn网络
        rnnOutput1, hn1 = self.gru1(prime, h0)  # rnnOutput 12,20,50 hn 1,20,50
        hn1 = hn1.view(batch_size, self.rnnhidden_dim)
        #hn1 = self.drop(hn1)
        fcOutput1 = self.fc3(hn1)
        #fcOutput1 = self.resrnn1.forward(prime, batch_size=batch_size)

        # 预测残差rnn网络
        rnnOutput2, hn2 = self.gru2(residual, h0)  # rnnOutput 12,20,50 hn 1,20,50
        hn2 = hn2.view(batch_size, self.rnnhidden_dim)
        #hn2 = self.drop(hn2)
        fcOutput2 = self.fc4(hn2)
        #fcOutput2 = self.resrnn2.forward(prime, batch_size=batch_size)

        # 合并预测结果
        result = fcOutput1+fcOutput2

        return result, fcOutput1, fcOutput2, residual


