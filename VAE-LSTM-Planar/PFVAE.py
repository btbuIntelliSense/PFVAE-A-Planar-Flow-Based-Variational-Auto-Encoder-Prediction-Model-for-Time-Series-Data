''' G66's  '''

import math
import time
import random
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.utils.data
import torch.optim as optim

import models.VAE as VAE
from utils.load_data import load_dataset
from optimization.loss import calculate_loss
from sklearn.metrics import mean_squared_error,mean_absolute_error

parser = argparse.ArgumentParser(description='基于标准化流的VAE，用于时序数据预测')

parser.add_argument('-d', '--dataset', type=str, default='PM2.5', choices=['PM2.5'],
                    metavar='DATASET', help='Dataset choice.')
parser.add_argument('-nc', '--no_cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--manual_seed', type=int,default=1, help='manual seed, if not given resorts to random seed.')
parser.add_argument('-li', '--log_interval', type=int, default=10, metavar='LOG_INTERVAL',
                    help='how many batches to wait before logging training status')

# optimization settings
parser.add_argument('-e', '--epochs', type=int, default=100, metavar='EPOCHS',
                    help='number of epochs to train (default: 2000)')
parser.add_argument('-bs', '--batch_size', type=int, default=30, metavar='BATCH_SIZE',
                    help='input batch size for training (default: 100)')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, metavar='LEARNING_RATE',
                    help='learning rate')
parser.add_argument('-f', '--flow', type=str, default='planar', choices=['planar', 'iaf', 'no_flow'],
                    help="""Type of flows to use, no flows can also be selected""")
parser.add_argument('-nf', '--num_flows', type=int, default=5,
                    metavar='NUM_FLOWS', help='Number of flow layers, ignored in absence of flows')
parser.add_argument('-mhs', '--made_h_size', type=int, default=24,
                    metavar='MADEHSIZE', help='Width of mades for iaf. Ignored for all other flows.')
parser.add_argument('--z_size', type=int, default=24, metavar='ZSIZE',
                    help='how many stochastic hidden units')
# gpu/cpu
parser.add_argument('--gpu_num', type=int, default=0, metavar='GPU', help='choose GPU to run on.')
'''
PM2.5
Namespace(batch_size=30, cuda=True, dataset='PM2.5', epochs=100, flow='planar', gpu_num=0, 
learning_rate=0.01, log_interval=1, made_h_size=24, manual_seed=1, no_cuda=False, num_flows=5, testing=True, z_size=24)
RMSE 24.456706506888096 
'''
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# 设置随机种子
random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
np.random.seed(args.manual_seed)

if args.cuda:
    # gpu device number
    torch.cuda.set_device(args.gpu_num)

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

def run(args, kwargs):
    print('\nMODEL SETTINGS: \n', args, '\n')
    print("Random Seed: ", args.manual_seed)

    # ==================================================================================================================
    # load_data
    # ==================================================================================================================
    train_loader,test_loader,x_train,y_train,x_test,y_test,scaler_1, scaler_2, args = load_dataset(args, **kwargs)

    # ==================================================================================================================
    # Select Model
    # ==================================================================================================================
    if args.flow == 'no_flow':
        model = VAE.VAE(args)
    elif args.flow == 'planar':
        model = VAE.PlanarVAE(args)
    elif args.flow == 'iaf':
        model = VAE.IAFVAE(args)
    else:
        raise ValueError('Invalid flow choice')

    if args.cuda:
        print("Model on GPU")
        model.cuda()
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    print('1 / x_train.shape[0]',1 / x_train.shape[0])
    beta = 1 / x_train.shape[0]

    # ==================================================================================================================
    # Training
    # ==================================================================================================================
    model.train()
    for epoch in range(1, args.epochs + 1):
        t_start = time.time()
        num_data = 0
        for batch_idx, (train_data, train_label) in enumerate(train_loader):

            if args.cuda:
                train_data = train_data.cuda()
                train_label = train_label.cuda()

            optimizer.zero_grad()
            train_x_mean, z_mu, z_var, ldj, z0, zk = model(train_data)
            # print('train_label', train_label.shape)
            # print('train_x_mean',train_x_mean.shape)
            loss, rec, kl, bpd = calculate_loss(train_x_mean, train_label, z_mu, z_var, z0, zk, ldj,args,beta)
            loss.backward()
            optimizer.step()

            rec = rec.item()
            kl = kl.item()
            num_data += len(train_data)

            if batch_idx % args.log_interval == 0:
                if args.input_type == 'binary':
                    print(
                        'Epoch: {:3d} [{:5d}/{:5d} ({:2.0f}%)]  \tLoss: {:11.6f}\trec: {:11.6f}\tkl: {:11.6f}'.format(
                            epoch, num_data, len(train_loader.sampler), 100. * batch_idx / len(train_loader),loss.item(), rec, kl))
        print('One training epoch took %.2f seconds' % (time.time() - t_start))

    # ==================================================================================================================
    # Testing
    # ==================================================================================================================
    model.eval()
    if args.cuda:
        x_test = x_test.cuda()
    test_x_mean, z_mu, z_var, ldj, z0, zk = model(x_test) # predict
    # batch_loss, rec, kl, batch_bpd = calculate_loss(test_x_mean, x_test, z_mu, z_var, z0, zk, ldj, args)
    # loss += batch_loss.item()

    y_test = scaler_1.inverse_transform(y_test.cuda().data.cpu().numpy().reshape(-1, 1))
    test_x_mean = scaler_2.inverse_transform(test_x_mean.cuda().data.cpu().numpy().reshape(-1, 1))

    # ==================================================================================================================
    # Figure and Compute Error
    # ==================================================================================================================
    def figure_error(y_test, test_x_mean):
        plt.figure(figsize=(12, 4))
        plt.plot(y_test.reshape(-1, 1)[2000:3000])
        plt.plot(test_x_mean.reshape(-1, 1)[2000:3000])
        plt.legend(['real', 'predict'], loc='best')
        plt.show()

        RMSE = math.sqrt(mean_squared_error(y_test,test_x_mean))
        MSE = mean_squared_error(y_test.reshape(-1, 1), test_x_mean.reshape(-1, 1))
        MAE = mean_absolute_error(y_test.reshape(-1, 1), test_x_mean.reshape(-1, 1))
        SMAPE = 2.0 * np.mean(np.abs(y_test - test_x_mean) / (np.abs(y_test) + np.abs(test_x_mean)))
        R = np.corrcoef(y_test.T, test_x_mean.T)
        print('RMSE', RMSE,'MSE',MSE,'MAE',MAE,'SMAPE',SMAPE,'R',R)

    figure_error(y_test, test_x_mean)

    # ==================================================================================================================
    # Save Results
    # ==================================================================================================================
    def save_result(y_test, x_mean):
        y_test = pd.DataFrame(data=y_test.reshape(-1, 1))
        y_test.to_csv("result/y_test.csv", index=False, header=['y_test'])
        x_mean = pd.DataFrame(data=x_mean.reshape(-1, 1))
        x_mean.to_csv("result/x_mean.csv", index=False, header=['pre_feature'])
    # save_result(y_test, test_x_mean)

if __name__ == "__main__":

    run(args, kwargs)

