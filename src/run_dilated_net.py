"""
Project:    stock_prediction
File:       run_dilated_net.py
Created by: louise
On:         26/02/18
At:         3:45 PM
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
from matplotlib import cm
from itertools import repeat

from models.dilated_cnn import DilatedNet
from data import SP500


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    # Keep track of loss in tensorboard
    writer = SummaryWriter()

    # Parameters
    learning_rate = 0.001
    batch_size = 16
    display_step = 1000
    max_epochs = 1000
    symbols = ['GOOGL', 'AAPL', 'AMZN']# AAPL, 'GOOG', 'GOOGL', 'FB', 'AMZN']
    n_stocks = len(symbols)
    n_hidden1 = 128
    n_hidden2 = 128
    n_steps_encoder = 20  # time steps, length of time window
    n_output = n_stocks
    T = 30

    # training data
    dset = SP500('data/sandp500/individual_stocks_5yr',
                 symbols=symbols,
                 start_date='2013-01-01',
                 end_date='2013-07-31',
                 T=T,
                 step=1)
    train_loader = DataLoader(dset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=4,
                              pin_memory=True  # CUDA only
                              )
    x, y = train_loader.dataset[0]
    print(x.shape)
    # Network Parameters
    model = DilatedNet(num_securities=n_stocks, n_layers=2, T=T, training=True).cuda()
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=0.0)  # n
    scheduler_model = lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.0)

    # loss function
    criterion = nn.MSELoss(size_average=True).cuda()

    losses = []
    it = 0
    for i in range(max_epochs):
        loss_ = 0.
        predicted = []
        gt = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data = Variable(data.permute(0, 2, 1)).contiguous()
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
        if i % 20 == 0:
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

    # TEST
    predictions = np.zeros((len(train_loader.dataset.chunks), n_stocks))
    ground_tr = np.zeros((len(train_loader.dataset.chunks), n_stocks))
    batch_size_pred = 4
    # Create test data set
    dtest = SP500('data/sandp500/individual_stocks_5yr',
                 symbols=symbols,
                 start_date='2013-01-01',
                 end_date='2013-10-31',
                 T=T)
    test_loader = DataLoader(dtest,
                              batch_size=batch_size_pred,
                              shuffle=False,
                              num_workers=4,
                              pin_memory=True  # CUDA only
                              )


    # Create list of n_stocks lists for storing predictions and GT
    predictions = [[] for i in repeat(None, len(symbols))]
    gts = [[] for i in repeat(None, len(symbols))]
    k = 0
    # Predictions
    for batch_idx, (data, target) in enumerate(test_loader):
        data = Variable(data.permute(0, 2, 1)).contiguous()
        target = Variable(target.unsqueeze_(1))
        if use_cuda:
            data = data.cuda()
            target = target.cuda()
        if target.data.size()[0] == batch_size_pred:
            output = model(data)
            for k in range(batch_size_pred):
                s = 0
                for stock in symbols:
                    predictions[s].append(output.data[k, s])
                    gts[s].append(target.data[k, 0, s])
                    s += 1
            k += 1


    # Plot results
    # Convert lists to np array for plot, and rescaling to original data
    if len(symbols) == 1:
        pred = dtest.scaler.inverse_transform(np.array(predictions[0]).reshape((len(predictions[0]), 1)))
        gt = dtest.scaler.inverse_transform(np.array(gts[0]).reshape(len(gts[0]), 1))
    if len(symbols) == 2:  #TODO(Louise) automate this part
        pred = dtest.scaler.inverse_transform(np.column_stack((predictions[0], predictions[1])))
        gt = dtest.scaler.inverse_transform(np.column_stack((gts[0], gts[1])))
    if len(symbols) == 3:
        pred = dtest.scaler.inverse_transform(np.column_stack((predictions[0], predictions[1], predictions[2])))
        gt = dtest.scaler.inverse_transform(np.column_stack((gts[0], gts[1], gts[2])))

    x = np.array(range(pred.shape[0]))
    s = 0
    for stock in symbols:
        h = plt.figure()
        plt.plot(x, pred[:, s], label="predictions", color=cm.Blues(300))
        plt.plot(x, gt[:, s], label="true", color=cm.Blues(100))
        plt.title(stock)
        plt.xlabel("Time (2013-01-01 to 2013-10-31)")
        plt.ylabel("Stock Price")
        plt.legend()
        plt.show()
        s += 1
