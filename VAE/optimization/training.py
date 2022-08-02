from __future__ import print_function
import torch

from torch.autograd import Variable
from optimization.loss import calculate_loss
from utils.visual_evaluation import plot_reconstructions
from utils.log_likelihood import calculate_likelihood
import math
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.preprocessing import StandardScaler

def train(epoch, train_loader, model, opt, args):

    model.train()
    train_loss = np.zeros(len(train_loader))
    train_bpd = np.zeros(len(train_loader))

    num_data = 0

    # set warmup coefficient 设置预热系数
    beta = min([(epoch * 1.) / max([args.warmup, 1.]), args.max_beta])
    print('beta = {:5.4f}'.format(beta))
    for batch_idx, (data, label) in enumerate(train_loader):

        if args.cuda:
            data = data.cuda()

        # # if args.dynamic_binarization:
        # #     data = torch.bernoulli(data)
        #
        # # data = Variable(data)
        # # print('data.type',data.type)
        # print('x = data', data.shape) #torch.Size([100, 24, 1])

        opt.zero_grad()
        # print('train_data.shape', data.shape)
        x_mean, z_mu, z_var, ldj, z0, zk = model(data)
        # print('train_x_mean.shape', x_mean.shape)

        loss, rec, kl, bpd = calculate_loss(x_mean, data, z_mu, z_var, z0, zk, ldj, args, beta=beta)

        loss.backward()
        train_loss[batch_idx] = loss.item()
        train_bpd[batch_idx] = bpd

        opt.step()

        rec = rec.item()
        kl = kl.item()

        num_data += len(data)

        if batch_idx % args.log_interval == 0:
            if args.input_type == 'binary':
                print('Epoch: {:3d} [{:5d}/{:5d} ({:2.0f}%)]  \tLoss: {:11.6f}\trec: {:11.6f}\tkl: {:11.6f}'.format(
                    epoch, num_data, len(train_loader.sampler), 100. * batch_idx / len(train_loader),
                    loss.item(), rec, kl))
            else:
                perc = 100. * batch_idx / len(train_loader)
                tmp = 'Epoch: {:3d} [{:5d}/{:5d} ({:2.0f}%)] \tLoss: {:11.6f}\tbpd: {:8.6f}'
                print(tmp.format(epoch, num_data, len(train_loader.sampler), perc, loss.item(), bpd),
                      '\trec: {:11.3f}\tkl: {:11.6f}'.format(rec, kl))

    if args.input_type == 'binary':
        print('====> Epoch: {:3d} Average train loss: {:.4f}'.format(
            epoch, train_loss.sum() / len(train_loader)))
    else:
        print('====> Epoch: {:3d} Average train loss: {:.4f}, average bpd: {:.4f}'.format(
            epoch, train_loss.sum() / len(train_loader), train_bpd.sum() / len(train_loader)))

    return train_loss


def evaluate(data_loader, model, args, testing=False, file=None, epoch=0):
    model.eval()
    loss = 0.
    batch_idx = 0
    bpd = 0.

    if args.input_type == 'binary':
        loss_type = 'elbo'
    else:
        loss_type = 'bpd'

    for data, _ in data_loader:
        batch_idx += 1

        if args.cuda:
            data = data.cuda()
        with torch.no_grad():
            data = Variable(data)
        # data = data.view(-1, *args.input_size)
        # print('111data', data.shape)
        x_mean, z_mu, z_var, ldj, z0, zk = model(data)
        # print('111x_mean', x_mean.shape)

        batch_loss, rec, kl, batch_bpd = calculate_loss(x_mean, data, z_mu, z_var, z0, zk, ldj, args)

        bpd += batch_bpd
        loss += batch_loss.item()

        # PRINT RECONSTRUCTIONS
        # if batch_idx == 1 and testing is False:
    # plot_reconstructions(data, x_mean, batch_loss, loss_type, epoch, args)
    # RMSE = math.sqrt(mean_squared_error(data.cuda().data.cpu().numpy().reshape(-1, 1),                                            x_mean.cuda().data.cpu().numpy().reshape(-1, 1)))
    # print('RMSE', RMSE)
    loss /= len(data_loader)
    bpd /= len(data_loader)

    # Compute log-likelihood
    if testing:
        with torch.no_grad():
            test_data = Variable(data_loader.dataset.tensors[0])

        if args.cuda:
            test_data = test_data.cuda()

        print('Computing log-likelihood on test set')

        model.eval()

        if args.dataset == 'caltech':
            log_likelihood, nll_bpd = calculate_likelihood(test_data, model, args, S=200, MB=24)
        else:
            log_likelihood, nll_bpd = calculate_likelihood(test_data, model, args, S=480, MB=24)
    else:
        log_likelihood = None
        nll_bpd = None

    if args.input_type in ['multinomial']:
        bpd = loss / (np.prod(args.input_size) * np.log(2.))

    if file is None:
        if testing:
            print('====> Test set loss: {:.4f}'.format(loss))
            print('====> Test set log-likelihood: {:.4f}'.format(log_likelihood))

            if args.input_type != 'binary':
                print('====> Test set bpd (elbo): {:.4f}'.format(bpd))
                print('====> Test set bpd (log-likelihood): {:.4f}'.format(log_likelihood/
                                                                           (np.prod(args.input_size) * np.log(2.))))

        else:
            print('====> Validation set loss: {:.4f}'.format(loss))
            if args.input_type in ['multinomial']:
                print('====> Validation set bpd: {:.4f}'.format(bpd))
    else:
        with open(file, 'a') as ff:
            if testing:
                print('====> Test set loss: {:.4f}'.format(loss), file=ff)
                print('====> Test set log-likelihood: {:.4f}'.format(log_likelihood), file=ff)

                if args.input_type != 'binary':
                    print('====> Test set bpd: {:.4f}'.format(bpd), file=ff)
                    print('====> Test set bpd (log-likelihood): {:.4f}'.format(log_likelihood /
                                                                               (np.prod(args.input_size) * np.log(2.))),
                          file=ff)

            else:
                print('====> Validation set loss: {:.4f}'.format(loss), file=ff)
                if args.input_type != 'binary':
                    print('====> Validation set bpd: {:.4f}'.format(loss / (np.prod(args.input_size) * np.log(2.))),
                          file=ff)

    if not testing:
        return loss, bpd
    else:
        return log_likelihood, nll_bpd

def evaluate2(data_loader, model,x_test,y_test,scaler_1, scaler_2, args, testing=False, file=None, epoch=0):
    model.eval()
    loss = 0.
    batch_idx = 0
    bpd = 0.

    if args.input_type == 'binary':
        loss_type = 'elbo'
    else:
        loss_type = 'bpd'

    # for index,(data, label) in enumerate(data_loader): # test_loader

    if args.cuda:
        data = x_test.cuda()
    # with torch.no_grad():
    #     data = Variable(x_test)
    # data = data.view(-1, *args.input_size) # x_test

    x_mean, z_mu, z_var, ldj, z0, zk = model(data) # predict
    batch_loss, rec, kl, batch_bpd = calculate_loss(x_mean, data, z_mu, z_var, z0, zk, ldj, args)
    bpd += batch_bpd
    loss += batch_loss.item()

    # PRINT RECONSTRUCTIONS
    # if batch_idx == 1 and testing is False:
    y_test = scaler_1.inverse_transform(y_test.cuda().data.cpu().numpy().reshape(-1, 1))
    x_mean = scaler_2.inverse_transform(x_mean.cuda().data.cpu().numpy().reshape(-1, 1))

    plot_reconstructions(y_test, x_mean, batch_loss, loss_type, epoch, args)
    RMSE = math.sqrt(mean_squared_error(y_test,x_mean))
    print('RMSE', RMSE)
    loss /= len(data_loader)
    bpd /= len(data_loader)

    # Compute log-likelihood
    if testing:
        with torch.no_grad():
            test_data = Variable(data_loader.dataset.tensors[0])

        if args.cuda:
            test_data = test_data.cuda()

        print('Computing log-likelihood on test set')

        model.eval()

        if args.dataset == 'caltech':
            log_likelihood, nll_bpd = calculate_likelihood(test_data, model, args, S=200, MB=24)
        else:
            log_likelihood, nll_bpd = calculate_likelihood(test_data, model, args, S=480, MB=24)
    else:
        log_likelihood = None
        nll_bpd = None

    if args.input_type in ['multinomial']:
        bpd = loss / (np.prod(args.input_size) * np.log(2.))

    if file is None:
        if testing:
            print('====> Test set loss: {:.4f}'.format(loss))
            print('====> Test set log-likelihood: {:.4f}'.format(log_likelihood))

            if args.input_type != 'binary':
                print('====> Test set bpd (elbo): {:.4f}'.format(bpd))
                print('====> Test set bpd (log-likelihood): {:.4f}'.format(log_likelihood/
                                                                           (np.prod(args.input_size) * np.log(2.))))

        else:
            print('====> Validation set loss: {:.4f}'.format(loss))
            if args.input_type in ['multinomial']:
                print('====> Validation set bpd: {:.4f}'.format(bpd))
    else:
        with open(file, 'a') as ff:
            if testing:
                print('====> Test set loss: {:.4f}'.format(loss), file=ff)
                print('====> Test set log-likelihood: {:.4f}'.format(log_likelihood), file=ff)

                if args.input_type != 'binary':
                    print('====> Test set bpd: {:.4f}'.format(bpd), file=ff)
                    print('====> Test set bpd (log-likelihood): {:.4f}'.format(log_likelihood /
                                                                               (np.prod(args.input_size) * np.log(2.))),
                          file=ff)

            else:
                print('====> Validation set loss: {:.4f}'.format(loss), file=ff)
                if args.input_type != 'binary':
                    print('====> Validation set bpd: {:.4f}'.format(loss / (np.prod(args.input_size) * np.log(2.))),
                          file=ff)

    if not testing:
        return loss, bpd
    else:
        return log_likelihood, nll_bpd
