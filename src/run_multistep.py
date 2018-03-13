"""
Project:    stock_prediction
File:       run_multistep.py
Created by: louise
On:         12/03/18
At:         4:40 PM
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
import matplotlib.dates as mdates
from matplotlib.dates import MonthLocator, DateFormatter
from itertools import repeat

from models.dilated_cnn import DilatedNet2DMultistep, DilatedNet2D
from data import SP500Multistep


# plot the forecasts in the context of the original dataset
def plot_forecasts(gt, predicted, n_test):
    """

    :param gt:
    :param predicted:
    :param n_test:
    :return:
    """
    predicted = np.array(predicted)
    gt = np.array(gt)
    # plot the forecasts in different colors
    for i in range(predicted.shape[0]):
        off_s = len(series) - n_test + i - 1
        off_e = off_s + len(forecasts[i]) + 1
        xaxis = [x for x in range(off_s, off_e)]
        yaxis = [series.values[off_s]] + forecasts[i]
        plt.plot(xaxis, yaxis)

    predicted = np.array(predicted)
    gt = np.array(gt)
    x = np.array(range(predicted.shape[0]))
    h = plt.figure()
    plt.plot(x, predicted[:, 0], label="predictions")
    plt.plot(x, gt[:, 0], label="true")
    plt.legend()
    plt.show()
    # show the plot
    plt.show()


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    # Keep track of loss in tensorboard
    writer = SummaryWriter()

    # Parameters
    learning_rate = 0.001
    batch_size = 16
    display_step = 100
    max_epochs = 500
    symbols = ['GOOGL', 'AAPL', 'AMZN', 'FB', 'ZION', 'NVDA', 'GS']
    n_stocks = len(symbols)
    n_hidden1 = 128
    n_hidden2 = 128
    n_steps_encoder = 20  # time steps, length of time window
    n_output = n_stocks
    T = 5
    start_date = '2013-01-01'
    end_date = '2013-12-31'
    n_step_data = 5
    n_out = 5
    n_in = 15

    fn_base = "multi_step_nstocks_" + str(n_stocks) + "_epochs_" + str(max_epochs) + "_T_" + str(T) + "_train_" + start_date + \
              "_" + end_date

    print(fn_base)

    # training data
    dset = SP500Multistep('data/sandp500/individual_stocks_5yr',
                          symbols=symbols,
                          start_date=start_date,
                          end_date=end_date,
                          step=n_step_data,
                          n_in=n_in,
                          n_out=5)
    train_loader = DataLoader(dset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=4,
                              pin_memory=True  # CUDA only
                              )
    x, y = train_loader.dataset[0]
    print(x)
    # Network Parameters
    model = DilatedNet2DMultistep(num_securities=n_stocks, T=T, training=True, n_in=T, n_out=n_out).cuda()
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=0.1)  # n
    scheduler_model = lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.)

    # loss function
    criterion = nn.MSELoss(size_average=False).cuda()

    losses = []
    it = 0
    for i in range(max_epochs):
        loss_ = 0.
        predicted = []
        gt = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data = Variable(data).unsqueeze_(1).contiguous()
            target = Variable(target.unsqueeze_(1))
            if use_cuda:
                data = data.cuda()
                target = target.cuda()
            optimizer.zero_grad()
            if target.data.size()[0] == batch_size:
                output = model(data)
                loss = criterion(output, target)
                loss_ += loss.data[0]
                loss.backward()
                optimizer.step()
                for k in range(batch_size):
                    predicted.append(output.data[k, 0, :, :].cpu().numpy())
                    gt.append(target.data[k, 0, :, :].cpu().numpy())
            it += 1

        print("Epoch = ", i)
        print("Loss = ", loss_)
        losses.append(loss_)
        writer.add_scalar("loss_epoch", loss_, i)

        scheduler_model.step()
        # Plot current predictions
        if i % display_step == 0:
            predicted = np.array(predicted)
            gt = np.array(gt)
            x = np.array(range(predicted.shape[0]))
            h = plt.figure()
            plt.plot(x, predicted[:, 0], label="predictions", color=cm.Blues(300))
            plt.plot(x, gt[:, 0], label="true", color=cm.Blues(100))
            plt.legend()
            plt.show()

    torch.save(model, 'conv2d_' + fn_base + '.pkl')

    h = plt.figure()
    x = xrange(len(losses))
    plt.plot(x, np.array(losses), label="loss")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.savefig("loss_" + fn_base + '.png')
    plt.legend()
    plt.show()

    # TEST
    predictions = np.zeros((len(train_loader.dataset.chunks), n_stocks))
    ground_tr = np.zeros((len(train_loader.dataset.chunks), n_stocks))
    batch_size_pred = 4
    #symbols = ['GOOGL', 'AAPL', 'AMZN', 'FB', 'ZION', 'NVDA', 'GS']

    # Create test data set
    start_date = '2013-01-01'
    end_date = '2017-10-31'
    dtest = SP500Multistep('data/sandp500/individual_stocks_5yr',
                           symbols=symbols,
                           start_date=start_date,
                           end_date=end_date,
                           step=n_step_data,
                           n_in=n_in,
                           n_out=n_out
                           )
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
        data = Variable(data.permute(0, 2, 1)).unsqueeze_(1).contiguous()
        target = Variable(target.unsqueeze_(1))
        if use_cuda:
            data = data.cuda()
            target = target.cuda()
        k = 0
        if target.data.size()[0] == batch_size_pred:
            output = model(data)
            for i in range(batch_size_pred):
                s = 0
                for stock in symbols:
                    predictions[s].append(output.data[i, 0, s, -n_out:])
                    gts[s].append(target.data[i, 0, s, -n_out:])
                    s += 1
                k += 1

    # Plot results
    # Convert lists to np array for plot, and rescaling to original data
    if len(symbols) == 1:
        pred = dtest.scaler.inverse_transform(np.array(predictions[0]).reshape((len(predictions[0]), 1)))
        gt = dtest.scaler.inverse_transform(np.array(gts[0]).reshape(len(gts[0]), 1))
    if len(symbols) >= 2:
        p = np.array(predictions)
        pred = dtest.scaler.inverse_transform(np.array(predictions).transpose())
        gt = dtest.scaler.inverse_transform(np.array(gts).transpose())

    x = [np.datetime64(start_date) + np.timedelta64(x, 'D') for x in range(0, pred.shape[0])]
    x = np.array(x)
    months = MonthLocator(range(1, 10), bymonthday=1, interval=3)
    monthsFmt = DateFormatter("%b '%y")
    s = 0
    for stock in symbols:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        plt.plot(x, pred[:, s], label="predictions", color=cm.Blues(300))
        plt.plot(x, gt[:, s], label="true", color=cm.Blues(100))
        ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
        ax.xaxis.set_major_locator(months)
        ax.xaxis.set_major_formatter(monthsFmt)
        plt.title(stock)
        plt.xlabel("Time")
        plt.ylabel("Stock Price")
        plt.legend()
        fig.autofmt_xdate()
        plt.savefig(stock + "_" + fn_base + '.png')
        plt.show()
        s += 1
