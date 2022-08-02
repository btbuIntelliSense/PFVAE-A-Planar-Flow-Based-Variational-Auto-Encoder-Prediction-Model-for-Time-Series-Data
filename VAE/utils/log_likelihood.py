from __future__ import print_function
import numpy as np
from scipy.special import logsumexp
from optimization.loss import calculate_loss_array


def calculate_likelihood(X, model, args, S=5000, MB=500):

    # set auxiliary variables for number of training and test sets
    # 为训练集和测试集的数量设置辅助变量
    N_test = X.size(0)

    X = X.view(-1, *args.input_size)
    # print('X.shape',X.shape) # torch.Size([110, 24, 1]) 测试集的形状
    likelihood_test = []

    if S <= MB:
        R = 1
    else:
        R = S // MB # 10
        S = MB # 24

    for j in range(N_test):
        if j % 100 == 0:
            print('Progress: {:.2f}%'.format(j / (1. * N_test) * 100))

        x_single = X[j].unsqueeze(0)
        # print('x_single.shape',x_single.shape) # torch.Size([1, 24, 1])

        a = []
        for r in range(0, R):
            # Repeat it for all training points
            # 对所有的训练点重复它
            x = x_single.expand(S, *x_single.size()[1:]).contiguous()
            # print('x.shape', x.shape) # torch.Size([24, 24, 1])

            x_mean, z_mu, z_var, ldj, z0, zk = model(x)
            # print('x_mean',x_mean.shape) # torch.Size([24, 24, 1])
            # print('z_mu',z_mu.shape) # torch.Size([24, 24, 24])
            # print('z_var',z_var.shape)# torch.Size([24, 24, 24])
            # print('ldj',ldj.shape)# torch.Size([24, 24, 24])
            # print('z0',z0.shape)# torch.Size([24, 24, 24])
            # print('zk', zk.shape)# torch.Size([24, 24, 24])

            a_tmp = calculate_loss_array(x_mean, x, z_mu, z_var, z0, zk, ldj, args)
            # print('a_tem.sahpe', a_tmp.shape) # torch.Size([24, 24])

            a.append(-a_tmp.cpu().data.numpy())

        # calculate max
        a = np.asarray(a)
        # print('a.shape',a.shape) # a.shape (20, 24, 24)
        a = np.reshape(a, (a.shape[0]* a.shape[1], 1))
        likelihood_x = logsumexp(a)
        likelihood_test.append(likelihood_x - np.log(len(a)))

    likelihood_test = np.array(likelihood_test)

    nll = -np.mean(likelihood_test)

    if args.input_type == 'multinomial':
        bpd = nll/(np.prod(args.input_size) * np.log(2.))
    elif args.input_type == 'binary':
        bpd = 0.
    else:
        raise ValueError('invalid input type!')

    return nll, bpd
