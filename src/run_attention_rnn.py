"""
Project:    stock_prediction
File:       run_attention_rnn.py
Created by: louise
On:         08/02/18
At:         12:53 PM
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

from models.dual_attention_rnn import DualAttentionRNN
from data import SP500


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    # Keep track of loss in tensorboard
    writer = SummaryWriter()

    # Parameters
    learning_rate = 0.001
    batch_size = 16
    display_step = 100
    max_epochs = 10
    symbols = ['AAPL', 'AMZN']#, AAPL, 'GOOG', 'GOOGL', 'FB', 'AMZN']
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
    model = DualAttentionRNN(n_stocks, encoder_hidden_dim=n_hidden1, decoder_hidden_dim=64, T=T).cuda()
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=0.0)  # n
    scheduler_model = lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.0)

    # loss function
    criterion = nn.MSELoss(size_average=False).cuda()

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
                output = model(data, data)
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

    # Check
    predictions = np.zeros((len(train_loader.dataset.chunks), n_stocks))
    ground_tr = np.zeros((len(train_loader.dataset.chunks), n_stocks))
    batch_size_pred = 4
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
    k = 0
    predicted1 = []
    predicted2 = []
    gt1 = []
    gt2 = []
    for batch_idx, (data, target) in enumerate(test_loader):
        data = Variable(data.permute(1, 0, 2)).contiguous()
        target = Variable(target.unsqueeze_(1))
        if use_cuda:
            data = data.cuda()
            target = target.cuda()
        if target.data.size()[0] == batch_size_pred:
            output = model(data)
            for k in range(batch_size_pred):
                #print(output.data[0, k])
                predicted1.append(output.data[k, 0])
                predicted2.append(output.data[k, 1])
                #print(target.data[0, :, k])
                gt1.append(target.data[k, 0, 0])
                gt2.append(target.data[k, 0, 1])
            k+=1


    # Plot results
    pred = np.column_stack((predicted1, predicted2))
    gt = np.column_stack((gt1, gt2))
    x = np.array(range(pred.shape[0]))
    h = plt.figure()
    plt.plot(x, pred[:, 0], label="predictions")
    #plt.plot(x, dtest.numpy_data[T-1:-1, 0], label='zizi')
    plt.plot(x, gt[:, 0], label="true")
    plt.legend()
    plt.show()
    h = plt.figure()
    plt.plot(x, pred[:, 1], label="predictions")
    plt.plot(x, gt[:, 1], label="true")
    plt.show()
    # h = plt.figure()
    # plt.plot(x, predictions[:, 2], label="predictions")
    # plt.plot(x, ground_tr[:, 2], label="true")
    # plt.show()
    # h = plt.figure()
    # plt.plot(x, predictions[:, 3], label="predictions")
    # plt.plot(x, ground_tr[:, 3], label="true")
    # plt.show()
