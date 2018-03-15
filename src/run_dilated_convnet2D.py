"""
Project:    stock_prediction
File:       run_dilated_convnet2D.py
Created by: louise
On:         26/02/18
At:         5:34 PM
"""
from itertools import repeat
# torch imports
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import optim
from torch.optim import lr_scheduler
# Tensorboard import
from tensorboardX import SummaryWriter
# Numpy, Matplotlib imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.dates as mdates
from matplotlib.dates import MonthLocator, DateFormatter
# Local imports
from models.dilated_cnn import DilatedNet2D
from data import SP500


if __name__ == "__main__":
    # Use cuda if available
    use_cuda = torch.cuda.is_available()
    # Keep track of loss in tensorboard
    writer = SummaryWriter()

    # Parameters
    learning_rate = 0.001
    batch_size = 16
    display_step = 200
    max_epochs = 1000
    symbols = ['GOOGL', 'AAPL', 'AMZN', 'FB', 'ZION', 'NVDA', 'GS']
    n_stocks = len(symbols)
    n_hidden1 = 128
    n_hidden2 = 128
    n_steps_encoder = 20  # time steps, length of time window
    n_output = n_stocks
    T = 20
    start_date = '2013-01-01'
    end_date = '2013-12-31'
    n_step_data = 10
    weight_decay = 0.01

    fn_base = "nstocks_" + str(n_stocks) + "_epochs_" + str(max_epochs) + "_T_" + str(T) + "_weight_decay_" + \
              str(weight_decay) + "_train_" + start_date + \
              "_" + end_date

    # Training data
    dset = SP500('data/sandp500/individual_stocks_5yr',
                 symbols=symbols,
                 start_date=start_date,
                 end_date=end_date,
                 T=T,
                 step=n_step_data)
    train_loader = DataLoader(dset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=use_cuda  # CUDA only
                              )

    # Network Definition + Optimizer + Scheduler
    model = DilatedNet2D(T=T,  hidden_size=64, dilation=2)
    if use_cuda:
        model = model.cuda()
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler_model = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    # loss function
    criterion = nn.MSELoss(size_average=False).cuda()
    # Store successive losses
    losses = []
    for i in range(max_epochs):
        loss_ = 0.
        predicted = []
        gt = []
        # Go through training data set
        for batch_idx, (data, target) in enumerate(train_loader):
            data = Variable(data.permute(0, 2, 1)).unsqueeze_(1).contiguous()
            target = Variable(target.unsqueeze_(1))
            if use_cuda:
                data = data.cuda()
                target = target.cuda()
            if target.data.size()[0] == batch_size:
                # Set gradient of optimizer to 0
                optimizer.zero_grad()
                # Compute predictions
                output = model(data)
                # Compute loss
                loss = criterion(output, target)
                loss_ += loss.data[0]
                # Backpropagation
                loss.backward()
                # Gradient descent step
                optimizer.step()
                # Store current results for visual check
                for k in range(batch_size):
                    predicted.append(output.data[k, 0, :].cpu().numpy())
                    gt.append(target.data[k, 0, :].cpu().numpy())

        print("Epoch = ", i)
        print("Loss = ", loss_)
        losses.append(loss_)
        # Store for display in Tensorboard
        writer.add_scalar("loss_epoch", loss_, i)
        # Apply step of scheduler for learning rate change
        scheduler_model.step()
        # Plot current predictions
        if i % display_step == 0:
            predicted = np.array(predicted)
            gt = np.array(gt)
            x = np.array(range(predicted.shape[0]))
            h = plt.figure()
            plt.plot(x, predicted[:, 0], label="predictions")
            plt.plot(x, gt[:, 0], label="true")
            plt.legend()
            plt.show()

    # Save trained models
    torch.save(model, 'conv2d_' + fn_base + '.pkl')
    # Plot training loss
    h = plt.figure()
    x = range(len(losses))
    plt.plot(np.array(x), np.array(losses), label="loss")
    plt.xlabel("Time")
    plt.ylabel("Training loss")
    plt.savefig("loss_" + fn_base + '.png')
    plt.legend()
    plt.show()

    ##########################################################################################
    # TEST
    predictions = np.zeros((len(train_loader.dataset.chunks), n_stocks))
    ground_tr = np.zeros((len(train_loader.dataset.chunks), n_stocks))
    batch_size_pred = batch_size

    # Create test data set
    start_date = '2013-01-01'
    end_date = '2017-10-31'
    dtest = SP500('data/sandp500/individual_stocks_5yr',
                  symbols=symbols,
                  start_date=start_date,
                  end_date=end_date,
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
                    predictions[s].append(output.data[i, 0, s])
                    gts[s].append(target.data[i, 0, s])
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
    # Plot for all stocks in
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
