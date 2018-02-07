"""
Project:    stock_prediction
File:       run.py
Created by: louise
On:         02/02/18
At:         2:22 PM
"""
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import optim
from torch.optim import lr_scheduler

from tensorboardX import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt

from lstm import LSTM
from data import SP500


class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future=0):
        outputs = []
        h_t = Variable(torch.zeros(input.size(0), 51).double(), requires_grad=False)
        c_t = Variable(torch.zeros(input.size(0), 51).double(), requires_grad=False)
        h_t2 = Variable(torch.zeros(input.size(0), 51).double(), requires_grad=False)
        c_t2 = Variable(torch.zeros(input.size(0), 51).double(), requires_grad=False)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):  # if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    # Keep track of loss in tensorboard
    writer = SummaryWriter()

    # Parameters
    learning_rate = 0.01
    batch_size = 16
    display_step = 100
    max_epochs = 50
    symbols = ['AAPL']#, 'GOOG', 'GOOGL', 'FB', 'AMZN']
    n_stocks = len(symbols)
    n_hidden1 = 128
    n_hidden2 = 128
    n_steps_encoder = 20  # time steps, length of time window
    n_output = n_stocks
    T = 10



    # training data
    dset = SP500('data/sandp500/individual_stocks_5yr',
                 symbols=symbols,
                 start_date='2013-01-01',
                 end_date='2013-03-31',
                 T=T)
    train_loader = DataLoader(dset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=4,
                              pin_memory=True  # CUDA only
                              )
    x, y = train_loader.dataset[0]
    print(x.shape)
    # Network Parameters
    model = LSTM(hidden_size=64, hidden_size2=300, num_securities=n_stocks, dropout=0.0, n_layers=10, T=T, training=True).cuda()
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=0.5)
    scheduler_model = lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.0)

    # loss function
    criterion = nn.SmoothL1Loss(size_average=True).cuda()

    losses = []
    it = 0
    for i in range(max_epochs):
        loss_ = 0.
        predicted = []
        gt = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data = Variable(data.permute(1, 0, 2)).contiguous()
            target = Variable(target.unsqueeze_(0))
            if use_cuda:
                data = data.cuda()
                target = target.cuda()
            optimizer.zero_grad()
            if target.data.size()[1] == batch_size:
                output = model(data)
                loss = criterion(output, target)
                loss_ += loss.data[0]
                loss.backward()
                optimizer.step()
                for k in range(batch_size):
                    predicted.append(output.data[k, 0])
                    gt.append(target.data[:, k, 0])
            it += 1

        print("Epoch = ", i)
        print("Loss = ", loss_)
        losses.append(loss_)
        writer.add_scalar("loss_epoch", loss_, i)

        scheduler_model.step()
        # Plot current predictions
        predicted = np.array(predicted).reshape(-1, 1)
        gt = np.array(gt).reshape(-1, 1)
        x = np.array(range(predicted.shape[0]))
        h = plt.figure()
        plt.plot(x, predicted[:, 0], label="predictions")
        plt.plot(x, gt[:, 0], label="true")
        plt.legend()
        plt.show()

    torch.save(model, 'prout.pkl')

    h = plt.figure()
    x = xrange(len(losses))
    plt.plot(x, np.array(losses), label="loss")
    plt.legend()
    plt.show()

    # Check
    predictions = np.zeros((len(train_loader.dataset.chunks), n_stocks))
    ground_tr = np.zeros((len(train_loader.dataset.chunks), n_stocks))
    batch_size_pred = 2
    dtest = SP500('data/sandp500/individual_stocks_5yr',
                 symbols=symbols,
                 start_date='2013-01-01',
                 end_date='2013-03-31',
                 T=T)
    test_loader = DataLoader(dtest,
                              batch_size=batch_size_pred,
                              shuffle=False,
                              num_workers=4,
                              pin_memory=True  # CUDA only
                              )
    k = 0
    predicted = []
    gt = []
    for batch_idx, (data, target) in enumerate(test_loader):
        data = Variable(data.permute(1, 0, 2)).contiguous()
        target = Variable(target.unsqueeze_(1))
        if use_cuda:
            data = data.cuda()
            target = target.cuda()
        if target.data.size()[0] == batch_size_pred:
            output = model(data)
            #print(output.data[0], target.data[0])
            predicted.append(output.data[0])
            predicted.append(output.data[1])
            gt.append(target.data[0])
            gt.append(target.data[1])
            # predictions[k*batch_size:((k+1)*batch_size), :] = dset.scaler.inverse_transform(output.data.cpu().numpy())
            # ground_tr[k*batch_size:((k+1)*batch_size), :] = dset.scaler.inverse_transform(target.squeeze_(1).data.cpu().numpy())
            #predictions[k * batch_size:((k + 1) * batch_size), :] = output.data.cpu().numpy()
            #ground_tr[k * batch_size:((k + 1) * batch_size), :] = target.squeeze_(1).data.cpu().numpy()
            k+=1


    # Plot results

    predicted = dset.scaler.inverse_transform(np.array(predicted).reshape(-1, 1))
    gt = dset.scaler.inverse_transform(np.array(gt).reshape(-1, 1))
    x = np.array(range(predicted.shape[0]))
    h = plt.figure()
    plt.plot(x, predicted[:, 0], label="predictions")
    #plt.plot(x, dtest.numpy_data[T-1:-1, 0], label='zizi')
    plt.plot(x, gt[:, 0], label="true")
    plt.legend()
    plt.show()
    # h = plt.figure()
    # plt.plot(x, predictions[:, 1], label="predictions")
    # plt.plot(x, ground_tr[:, 1], label="true")
    # plt.show()
    # h = plt.figure()
    # plt.plot(x, predictions[:, 2], label="predictions")
    # plt.plot(x, ground_tr[:, 2], label="true")
    # plt.show()
    # h = plt.figure()
    # plt.plot(x, predictions[:, 3], label="predictions")
    # plt.plot(x, ground_tr[:, 3], label="true")
    # plt.show()
