from __future__ import print_function

import torch
import torch.utils.data as data_utils
import pickle
from scipy.io import loadmat
import pandas as pd
import numpy as np

import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler

def load_Time_series_data(args, **kwargs):
    args.dynamic_binarization = False
    args.input_type = 'binary'

    args.input_size = [24, 1]

    scaler_1 = StandardScaler()# 不同的数据归一方式，计算出来的正则化项不同
    scaler_2 = StandardScaler() # MinMaxScaler()
    data1 = pd.read_csv('F:/66的研究生日子/科研生活/研一暑假学习/文档/程序-泓哥/data/guanyuan.csv').values.reshape(-1, 7)
    # data1 = pd.read_csv('F:/66的研究生日子/科研生活/研一暑假学习/文档/程序-泓哥/data/haidian_weather.csv').values.reshape(-1, 5)
    # data1 = pd.read_csv(r'F:\66的研究生日子\科研生活\研二 -- 上\论文2\data\shunyi.csv').values.reshape(-1, 7)
    label = data1[:, 5].reshape(-1, 1)
    data = np.array(data1[:, [5]]).reshape(-1, 1)

    def handle_data(scaler_1,scaler_2,data,label):
        scaler_1 = scaler_1
        scaler_2 = scaler_2
        data_scaled = scaler_1.fit_transform(data) #data #
        label_scaled = scaler_2.fit_transform(label)#label #
        print('data_scaled.shape：',data_scaled.shape)
        return data_scaled ,label_scaled, scaler_1, scaler_2
    data_scaled, label_scaled, scaler_1, scaler_2 = handle_data(scaler_1,scaler_2,data,label)
    def create_timestamps_ds(data,label,window_size,feature_num):
        features = data[:len(label) - window_size, 0:feature_num].reshape(-1, 24, feature_num)
        labels = label[window_size:len(label)].reshape(-1, 24)
        features = torch.tensor(features).float()
        labels = torch.tensor(labels).float()
        print('features.shape:',features.shape)
        print('labels.shape:',labels.shape)
        return features, labels
    features, labels = create_timestamps_ds(data_scaled, label_scaled,24,1)
    x_train, x_test, y_train, y_test = train_test_split(features,
                                                        labels,
                                                        test_size=0.2,
                                                        random_state=42,
                                                        shuffle=False, )
    print('x_train',x_train.shape) #torch.Size([983, 24, 1])
    print('y_train', y_train.shape)  # torch.Size([983, 24, 1])
    print('x_test.shape', x_test.shape)  # torch.Size([110, 24, 1])
    ds_train = torch.utils.data.TensorDataset(x_train, y_train)
    ds_test = torch.utils.data.TensorDataset(x_test, y_test)
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader,x_train,y_train,x_test,y_test, scaler_1, scaler_2, args

def load_dataset(args, **kwargs):

    if args.dataset == 'PM2.5': # shape(batch,24,1)
        train_loader, test_loader,x_train,y_train,x_test,y_test,scaler_1, scaler_2, args = load_Time_series_data(args, **kwargs)

    else:
        raise Exception('Wrong name of the dataset!')

    return train_loader, test_loader, x_train,y_train,x_test,y_test,scaler_1, scaler_2, args