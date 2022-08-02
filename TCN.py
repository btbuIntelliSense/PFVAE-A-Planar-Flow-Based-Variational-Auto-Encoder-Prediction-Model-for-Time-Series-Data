# from tensorflow.examples.tutorials.mnist import input_data
from keras.models import Model
from keras.layers import add, Input, Conv1D, Activation, Flatten, Dense

# -- coding: utf-8 --
__author__ = 'gwt'

import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,LSTM,Bidirectional,GRU
from keras import optimizers
from keras.optimizers import RMSprop
from keras.layers.convolutional import Conv2D,MaxPooling2D,Conv1D
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error #平方绝对误差

# from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split

np.random.seed(1)
random.seed(1)
tf.random.set_seed(1)

# data1 = pd.read_csv("F:/66的研究生日子/科研生活/研一暑假学习/文档/程序-泓哥/data/guanyuan.csv").values.reshape(-1, 7)
# data=data[['aqi','co', 'no2', 'o3', 'pm10','pm25','so2']]
# data1 = pd.read_csv("F:/66的研究生日子/科研生活/研一暑假学习/文档/程序-泓哥/data/haidian_weather.csv").values.reshape(-1, 5)
# data1 = pd.read_csv(r'F:\66的研究生日子\科研生活\研二 -- 上\论文2\data\shunyi.csv').values.reshape(-1, 7)
# data3 = np.concatenate((data1, data2), axis=1)
data1 = pd.read_csv(r'F:\66的研究生日子\科研生活\Power_load.csv').values.reshape(-1, 1)

window_size = 24
feature_num = 1

label = data1[:, 0].reshape(-1, 1)
data = np.array(data1[:, [0]]).reshape(-1, feature_num)

scaler_1 = StandardScaler()
scaler_2 = StandardScaler()

data_scaled = scaler_1.fit_transform(data)
label_scaled = scaler_2.fit_transform(label)

# 创建特征和标签数据对
def create_timestamps_ds1(data, label=label_scaled,
                          window_size=window_size, feature_num=feature_num):
    features = data[:len(label) - window_size, 0:feature_num].reshape(-1, 24, feature_num)
    labels = label[window_size:len(label)].reshape(-1, 24)

    # features = torch.tensor(features).float()
    # labels = torch.tensor(labels).float()
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


# 残差块
def ResBlock(x, filters, kernel_size, dilation_rate):
    r = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate, activation='relu')(x)  # 第一卷积
    r = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate)(r)  # 第二卷积
    if x.shape[-1] == filters:
        shortcut = x
    else:
        shortcut = Conv1D(filters, kernel_size, padding='same')(x)  # shortcut（捷径）
    o = add([r, shortcut])
    o = Activation('relu')(o)  # 激活函数
    return o


# 序列模型
def TCN(train_x, train_y, X_test, y_test):
    inputs = Input(shape=(24, 1))
    x = ResBlock(inputs, filters=3, kernel_size=5, dilation_rate=1) # 一层
    x = ResBlock(x, filters=5, kernel_size=3, dilation_rate=2)
    x = ResBlock(x, filters=3, kernel_size=3, dilation_rate=4)
    x = Flatten()(x)
    x = Dense(24)(x)
    model = Model(inputs, x)
    # 查看网络结构
    model.summary()
    # 编译模型
    opt = optimizers.Adam(lr=0.01)
    model.compile(loss='mae', optimizer=opt)  ## mae
    # 训练模型
    model.fit(train_x, train_y, batch_size=30, epochs=100, verbose=2)
    # 评估模型
    testPredict = model.predict(X_test)
    testPredict = scaler_2.inverse_transform(testPredict)
    y_test = scaler_2.inverse_transform(y_test)

    plt.figure(figsize=(12, 4))
    plt.plot(y_test.reshape(-1, 1)[2000:3000])
    plt.plot(testPredict.reshape(-1, 1)[2000:3000])
    plt.legend(['TCN_real', 'TCN_predict'], loc='best')
    plt.show()

    data4 = np.array(testPredict).reshape(-1, 1)
    data5 = np.array(y_test).reshape(-1, 1)
    RMSE = math.sqrt(mean_squared_error(testPredict.reshape(-1, 1), y_test.reshape(-1, 1)))  # 均方误差
    MSE = mean_squared_error(testPredict.reshape(-1, 1), y_test.reshape(-1, 1))
    MAE = mean_absolute_error(testPredict.reshape(-1, 1), y_test.reshape(-1, 1))
    SMAPE = 2.0 * np.mean(np.abs(y_test - testPredict) / (np.abs(y_test) + np.abs(testPredict))) * 100
    R = np.corrcoef(data4.T, data5.T)
    print('RMSE', RMSE,  'MAE', MAE, 'SMAPE', SMAPE,'MSE', MSE, 'R', R)

    # 保存结果
    def save_result(y_test, x_mean):
        y_test = pd.DataFrame(data=y_test.reshape(-1, 1))
        y_test.to_csv(r"F:\66的研究生日子\科研生活\研二 -- 上\论文2\code\VAE-LSTM-Planar\result\TCN_y_test.csv", index=False, header=['y_test'])
        x_mean = pd.DataFrame(data=x_mean.reshape(-1, 1))
        x_mean.to_csv(r"F:\66的研究生日子\科研生活\研二 -- 上\论文2\code\VAE-LSTM-Planar\result\TCN_test_pred.csv", index=False,
                      header=['pre_feature'])
    # save_result(y_test, testPredict)



# train_x, train_y, valid_x, valid_y, test_x, test_y = read_data('MNIST_data')
TCN(X_train, y_train, X_test, y_test)

