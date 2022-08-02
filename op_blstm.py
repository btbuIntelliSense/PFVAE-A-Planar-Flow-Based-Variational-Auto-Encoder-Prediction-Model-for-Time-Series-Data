
import torch
import torch.nn as nn
import torch.optim as optim
from blitz.modules import BayesianLSTM,BayesianLinear, BayesianGRU
from blitz.utils import variational_estimator

import os
import time
import math
import cvxpy # 凸优化库
import scipy
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
# start = time.clock()
torch.manual_seed(1)
np.random.seed(1)
plt.rcParams['font.sans-serif'] = [u'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def fn_n(params):
    print('### ******************************************************* ###')
    print ('Params testing:{} '.format(params))

    # path1 = 'F:/66的研究生日子/科研生活/研一暑假学习/文档/程序-泓哥/data/guanyuan.csv'
    # data1 = pd.read_csv(path1).values.reshape(-1, 7)

    # path2 = 'F:/66的研究生日子/科研生活/研一暑假学习/文档/程序-泓哥/data/haidian_weather.csv'
    # data2 = pd.read_csv(path2).values.reshape(-1, 5)
    # data2 = pd.read_csv(r'F:\66的研究生日子\科研生活\研二 -- 上\论文2\data\shunyi.csv').values.reshape(-1, 7)
    # data3 = np.concatenate((data1, data2), axis=1)

    data1 = pd.read_csv(r'F:\66的研究生日子\科研生活\Power_load.csv').values.reshape(-1, 1)

    window_size = 24
    feature_num = 1

    # data.head()
    # data_pm25 = np.array(data1["pm25"])
    # 归一化特征scaler_1:归一化特征；scaler_2：归一化标签
    # 选择训练和测试数据范围
    # data = np.array(data)[:,5].reshape(-1, 1)
    label = data1[:, 0].reshape(-1, 1)
    # data = np.array(data[:,[0,5,6,11]]).reshape(-1, 4)
    # np.array 创建多维数组
    data = np.array(data1[:,[0]]).reshape(-1, feature_num)

    scaler_1 =StandardScaler() #MinMaxScaler()#
    scaler_2 =StandardScaler()# MinMaxScaler()#StandardScaler()

    data_scaled = scaler_1.fit_transform(data)
    label_scaled = scaler_2.fit_transform(label)

    # 创建特征和标签数据对
    def create_timestamps_ds1(data, label=label,
                              window_size=window_size, feature_num=feature_num):
        features = data[:len(label) - window_size, 0:feature_num].reshape(-1, 24, feature_num)
        labels = label[window_size:len(label)].reshape(-1, 24)

        features = torch.tensor(features).float()
        labels = torch.tensor(labels).float()
        print(features.shape)
        print(labels.shape)
        return features, labels

    features, labels = create_timestamps_ds1(data_scaled, label_scaled)
    X_train, X_test, y_train, y_test = train_test_split(features,
                                                        labels,
                                                        test_size=.2,
                                                        random_state=42,
                                                        shuffle=False)

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    ds = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader_train = torch.utils.data.DataLoader(ds, batch_size=params['batch_size'], shuffle=False)

    # 变分优化器
    @variational_estimator
    class NN(nn.Module):
        def __init__(self):
            super(NN, self).__init__()
            # self.conv_1 = BayesianConv1d(5, 24, 10, 5)
            # self.conv_2 = BayesianConv2d(5, 24, (3, 3), 2)
            self.lstm_1 = BayesianLSTM(feature_num, params['units'], prior_sigma_1=10, posterior_rho_init=-10.0)
            self.linear = nn.Linear(24, 24)

        def forward(self, x):
            # x_non = torch.tensor(non_normalize_arctan(x, a), dtype=torch.float32)
            x_, _ = self.lstm_1(x)
            # gathering only the latent end-of-sequence for the linear layer
            x_ = x_[:, -1, :] # 这是什么意思？
            out = self.linear(x_)
            return out

    # 定义网络
    net = NN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=params['learn_rate'])

    # 网络训练
    iteration = 0
    epochs = params['epochs']
    for epoch in range(1,epochs + 1):
        print(epoch)
        since = time.time()
        for i, (datapoints, labels) in enumerate(dataloader_train):
            optimizer.zero_grad()

            loss = net.sample_elbo(inputs=datapoints,
                                   labels=labels,
                                   criterion=criterion,
                                   sample_nbr=1,
                                   complexity_cost_weight=1 / X_train.shape[0])
            loss.backward()

            optimizer.step()

            iteration += 1
            if iteration % 250 == 0:
                preds_test = net(X_test)[:, 0].unsqueeze(1)
                loss_test = criterion(preds_test, y_test)
                print("Iteration: {} Val-loss: {:.4f}".format(str(iteration), loss_test))
        Loss0 = torch.tensor(loss)
        # torch.save(Loss0,'loss/epoch_{}'.format(epoch))
        Loss_list = []
        Loss_list.append(loss/len(X_train[0]))
        time_elapsed = time.time() - since
        print('Loss_list:{},Training complete in {:.0f}m {:.0f}s'.format(Loss_list, time_elapsed // 60, time_elapsed % 60))
        # 网络测试
    def pred_future(X_test,
                    sample_nbr=1):
        # sorry for that, window_size is a global variable, and so are X_train and Xs
        global scaler
        preds_test = []
        test_deque = []
        # predict it and append to list
        for i in range(n):
            # print(i)
            #  print(X_test[i])
            as_net_input = torch.tensor(X_test[i]).unsqueeze(0) # unsqueeze：增加维度
            pred = [net(as_net_input).cpu().data.numpy() for i in range(sample_nbr)]
            test_deque.append(torch.tensor(pred).mean().cpu().item())
            preds_test.append(pred)

        return preds_test


    # 测试结果反归一化
    original = label
    # original = np.array(data1)[:, 0]
    print(original.shape)
    n = len(y_test)
    y_test = np.array(original[-n * 24:]).reshape(n, 24)
    print(y_test)

    # 结果分析
    sample_nbr = 1
    preds_test = pred_future(X_test, sample_nbr)

    preds_test = np.array(preds_test).reshape(n, sample_nbr, 24)
    print(type(preds_test))
    # 22 n 24
    print(preds_test.shape)
    new_preds_test = []
    new_preds_test_std = []

    std = []
    var = []
    std1 = []

    for i in range(n):
        if i == 21:
            print(preds_test[i])
        new_preds_test.append(np.mean(preds_test[i], axis=0))
        new_preds_test_std.append((scaler_2.inverse_transform(preds_test[i])))
        std.append(np.std(preds_test[i], axis=0))
        var.append(np.var(preds_test[i], axis=0))

    new_preds_test = np.array(new_preds_test)
    new_preds_test_std = np.array(new_preds_test_std)
    print("新std")
    print(new_preds_test_std.shape)

    for i in range(n):
        std1.append(np.std(new_preds_test_std[i], axis=0))

    std = np.array(std)
    std1 = np.array(std1)
    var = np.array(var)
    print(new_preds_test.shape)
    print(std.shape)
    print(var.shape)
    print(std1[0])
    print(new_preds_test[0])
    std = scaler_2.inverse_transform(std)
    new_preds_test = scaler_2.inverse_transform(new_preds_test)
    var = scaler_2.inverse_transform(var)
    print(new_preds_test[0])
    print(std[0])

    # print(time.clock() - start)
    data4 = np.array(new_preds_test).reshape(-1, 1)
    data5 = np.array(y_test).reshape(-1, 1)
    RMSE = math.sqrt(mean_squared_error(new_preds_test.reshape(-1, 1), y_test.reshape(-1, 1)))  # 均方误差
    MSE = mean_squared_error(new_preds_test.reshape(-1, 1), y_test.reshape(-1, 1))
    MAE = mean_absolute_error(new_preds_test.reshape(-1, 1), y_test.reshape(-1, 1))
    SMAPE = 2.0 * np.mean(np.abs(y_test.reshape(-1, 1) - new_preds_test.reshape(-1, 1)) / (np.abs(y_test.reshape(-1, 1)) + np.abs(new_preds_test.reshape(-1, 1)))) * 100
    R = np.corrcoef(data4.T, data5.T)
    print('RMSE', RMSE,  'MAE', MAE, 'SMAPE', SMAPE,'MSE', MSE, 'R', R)

    # 画图展示
    plt.figure(figsize=(12,4))
    plt.plot(y_test.reshape(-1, 1)[2000:3000], )
    plt.plot(new_preds_test.reshape(-1, 1)[2000:3000],)
    plt.legend(['Bayes_predict', 'Bayes_real'], loc='best')
    plt.show()

    def save_result(y_test, x_mean):
        y_test = pd.DataFrame(data=y_test.reshape(-1, 1))
        y_test.to_csv(r"..\code\VAE-LSTM-Planar\result\Bayes_y_test.csv", index=False, header=['y_test'])
        x_mean = pd.DataFrame(data=x_mean.reshape(-1, 1))
        x_mean.to_csv(r"..\code\VAE-LSTM-Planar\result\Bayes_x_mean.csv", index=False, header=['pre_feature'])
    # save_result(y_test, new_preds_test)

    acc = math.sqrt(mean_squared_error(y_test, new_preds_test))
    return {'loss': acc, 'status': STATUS_OK} ## 长记性！！！！！！！！！！！
# 定义贝叶斯优化的超参数的优化空间
parameter_space = {
    'test_size':hp.choice('test_size',[0.2]),
    'batch_size':hp.choice('batch_size',[30]),
    'units': hp.choice('units', [24]),
    'prior_sigma_1':hp.choice('prior_sigma_1',[10]),
    'posterior_rho_init':hp.choice('posterior_rho_init',[-10]),
    'learn_rate':hp.choice('learn_rate',[0.01]),#
    'epochs':hp.choice('epochs',[50]),
    'sample_nbr':hp.choice('sample_nbr',[1]),
    # 'nmv':hp.choice('nmv',[0.280706567865728])
}
'''
hp.choice(label, options) 其中options应是 python 列表或元组。
hp.normal(label, mu, sigma) 其中mu和sigma分别是均值和标准差。
hp.uniform(label, low, high) 其中low和high是范围的下限和上限。
内部可以这样定义："n_iter":hp.choice("n_iter",range(30,50))
'''

### ******************************************************* ###
# 调用贝叶斯优化
trials = Trials()
# f_nn:目标函数 space：搜索空间 algo：搜索算法 max_evals：指定fmin函数执行的最大次数max_evals trials：获取内部优化过程
best = fmin(fn=fn_n,space=parameter_space,algo=tpe.suggest,max_evals=1,trials=trials)
# space_eval(parameter_space, best) # 不用此函数，返回的best将是各参数的索引位置
print('best: ',space_eval(parameter_space, best))
