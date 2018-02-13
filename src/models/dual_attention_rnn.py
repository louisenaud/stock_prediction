"""

   Project : stock_prediction
   dual_attention_rnn.py created by Louise Naud
   On : 1/29/18
   At : 19:10
   
"""
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class InputAttentionEncoder(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, T=10, in_features=2*128 + 10-1, num_layers=1, out_features=1):
        """
        Constructor for Input Attention Encoder
        :param input_dim: int, number of time series in data set.
        :param hidden_dim: int, dimension of hidden layer
        :param T: int, horizon - length of series
        :param num_layers: int - number of layers in LSTM
        :param in_features: int, 2 * hidden_dim + T - 1
        :param out_features: int, dimension of output, 1 by default
        """
        super(InputAttentionEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.T = T

        self.lstm_unit = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers)
        self.attn_linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.fc2 = nn.Linear(out_features, out_features)

    def forward(self, input_batch):
        """
        Forward overload.
        :param input_batch: Pytorch Tensor, batch_size * T - 1 * input_dim
        :return:
        """
        input_weighted = Variable(input_batch.data.new(input_batch.size(0), self.T - 1, self.input_dim).zero_())
        input_encoded = Variable(input_batch.data.new(input_batch.size(0), self.T - 1, self.hidden_dim).zero_())
        # hidden, cell: initial states with dimension hidden_dim
        hidden = self.init_hidden(input_batch)  # 1 * batch_size * hidden_dim
        cell = self.init_hidden(input_batch)
        xx = input_batch[:, 0:self.T-1, :].contiguous()
        input_batch_f = input_batch.type(torch.cuda.FloatTensor)
        for t in range(self.T - 1):
            h = hidden.repeat(self.input_dim, 1, 1)
            c = cell.repeat(self.input_dim, 1, 1)
            xxx = xx.permute(0, 2, 1)
            # Concatenate hidden states with each predictor
            x = torch.cat((hidden.repeat(self.input_dim, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.input_dim, 1, 1).permute(1, 0, 2),
                           xx.permute(0, 2, 1)),
                          dim=2)  # batch_size * input_dim * (2*hidden_dim + T - 1)
            # Attention weights
            x = self.attn_linear(x)  # (batch_size * input_dim) * 1
            #x = self.bn1(x.view(self.input_dim, -1).type_as(x))
            attn_weights = F.softmax(x.view(-1, self.input_dim).type_as(x))  # batch_size * input_dim, attn weights with values sum up to 1.
            # Eqn. 10: LSTM
            weighted_input = torch.mul(attn_weights, input_batch_f[:, t, :]).type_as(x) # batch_size * input_dim
            self.lstm_unit.flatten_parameters()
            _, lstm_states = self.lstm_unit(weighted_input.unsqueeze(0), (hidden, cell))
            hidden = lstm_states[0]
            cell = lstm_states[1]
            # Save output
            input_weighted[:, t, :] = weighted_input
            input_encoded[:, t, :] = hidden
        return input_weighted.type_as(input_batch), input_encoded.type_as(input_batch)

    def init_hidden(self, input_batch):
        """
        Initialize hidden variable.
        :param x:
        :return:
        """
        return Variable(input_batch.data.new(1, input_batch.size(0), self.hidden_dim).zero_())


class TemporalAttentionDecoder(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim, T):
        """
        Constructor for Temporal Attention Decoder.
        :param encoder_hidden_dim: int
        :param decoder_hidden_dim: int
        :param T: int
        """
        super(TemporalAttentionDecoder, self).__init__()

        self.T = T
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim

        self.attn_layer = nn.Sequential(nn.Linear(2 * decoder_hidden_dim + encoder_hidden_dim, encoder_hidden_dim),
                                         nn.ReLU(), nn.Linear(encoder_hidden_dim, 1))
        self.lstm_layer = nn.LSTM(input_size=1, hidden_size=decoder_hidden_dim)
        self.fc = nn.Linear(encoder_hidden_dim + 1, 1)
        self.fc_final = nn.Linear(decoder_hidden_dim + encoder_hidden_dim, 1)

        self.fc.weight.data.normal_()

    def forward(self, input_encoded, y_history):
        """

        :param input_encoded: Pytorch Variable, batch_size * T - 1 * encoder_hidden_dim
        :param y_history: Pytorch Variable, batch_size * (T-1)
        :return: 1 * batch_size * decoder_hidden_dim
        """
        hidden = self.init_hidden(input_encoded)
        cell = self.init_hidden(input_encoded)
        # hidden.requires_grad = False
        # cell.requires_grad = False
        for t in range(self.T - 1):
            # Temporal Attention weights batch_size * T * (2*decoder_hidden_dim + encoder_hidden_dim)
            x = torch.cat((hidden.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.T - 1, 1, 1).permute(1, 0, 2), input_encoded), dim=2)
            #x = F.softmax(self.attn_layer(x.view(-1, 2 * self.decoder_hidden_dim + self.encoder_hidden_dim)).view(-1, self.T - 1)) # batch_size * T - 1, row sum up to 1
            x = F.softmax(self.attn_layer(x).view(-1, self.T -1))  # batch_size * T - 1, row sum up to 1
            # Context vector
            context = torch.bmm(x.unsqueeze(1), input_encoded)[:, 0, :] # batch_size * encoder_hidden_dim
            if t < self.T - 1:
                # Concatenate context vector and previous values of stock, pass through linear layer -> 1
                y_tilde = self.fc(torch.cat((context, y_history[:, t]), dim=-1))  # batch_size * 1
                # LSTM
                self.lstm_layer.flatten_parameters()
                _, lstm_output = self.lstm_layer(y_tilde.unsqueeze_(0), (hidden, cell))
                hidden = lstm_output[0]  # 1 * batch_size * decoder_hidden_dim
                cell = lstm_output[1]    # 1 * batch_size * decoder_hidden_dim
        # Estimated stock price
        y_pred = self.fc_final(torch.cat((hidden[0], context), dim=1))
        return y_pred

    def init_hidden(self, x):
        """

        :param x:
        :return:
        """
        return Variable(x.data.new(1, x.size(0), self.decoder_hidden_dim).zero_())


class DualAttentionRNN(nn.Module):
    def __init__(self, input_dim, encoder_hidden_dim=64, decoder_hidden_dim=64, T=10,
                 parallel=False):
        """

        :param input_dim: X.shape[1]
        :param encoder_hidden_dim:
        :param decoder_hidden_dim:
        :param T:
        :param parallel:
        """
        super(DualAttentionRNN, self).__init__()
        self.encoder = InputAttentionEncoder(input_dim, hidden_dim=encoder_hidden_dim, T=T)
        self.decoder = TemporalAttentionDecoder(encoder_hidden_dim=encoder_hidden_dim,
                                                decoder_hidden_dim=decoder_hidden_dim, T=T)

        if parallel:
            self.encoder = nn.DataParallel(self.encoder)
            self.decoder = nn.DataParallel(self.decoder)

    def forward(self, batch_x, y_history):
        """

        :param batch_x:
        :param y_history:
        :return:
        """
        input_weighted, input_encoded = self.encoder.forward(batch_x)
        y_pred = self.decoder.forward(input_encoded, y_history.type_as(batch_x))
        return y_pred


class MultipleStocksEmbedding(nn.Module):
    def __int__(self, number_stocks, embed_dimension, T=10):
        """

        :param number_stocks: int, number of different time series in dataset
        :param embed_dimension: int, embedding dimension embed_dimension << number_stocks
        :param T:
        :return:
        """
        self.T = T
        self.embed_layer = nn.Embedding(number_stocks, embed_dimension)

    def forward(self, batch_x):
        """

        :param batch_x:
        :return:
        """
        return self.embed_layer(batch_x)
