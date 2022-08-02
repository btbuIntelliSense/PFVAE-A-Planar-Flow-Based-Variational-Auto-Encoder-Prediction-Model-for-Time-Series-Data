import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from blitz.modules import BayesianLSTM, BayesianGRU
from blitz.utils import variational_estimator
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import mean_squared_error,mean_absolute_error
import time
plt.rcParams['font.sans-serif'] = [u'SimHei']
plt.rcParams['axes.unicode_minus'] = False
torch.manual_seed(1)
np.random.seed(1)

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

scaler_1 = StandardScaler()
scaler_2 = StandardScaler()

data_scaled = scaler_1.fit_transform(data)
label_scaled = scaler_2.fit_transform(label)

# 创建特征和标签数据对
def create_timestamps_ds1(data, label=label_scaled,
                          window_size=window_size, feature_num=feature_num):
    features = data[:len(label) - window_size, 0:feature_num].reshape(-1, 24, feature_num)
    labels = label[window_size:len(label)].reshape(-1, 24)

    features = torch.tensor(features).float()
    labels = torch.tensor(labels).float()
    print('feature.shape',features.shape)
    print('labels.shape',labels.shape)
    return features, labels

features, labels = create_timestamps_ds1(data_scaled, label_scaled)
X_train, X_test, y_train, y_test = train_test_split(features,
                                                    labels,
                                                    test_size=.2,
                                                    random_state=42,
                                                    shuffle=False)

print('x_train.shape',X_train.shape)
print('test.shape',X_test.shape)
print('y_train.shape',y_train.shape)
print('y_test.shape',y_test.shape)

ds = torch.utils.data.TensorDataset(X_train, y_train)
dataloader_train = torch.utils.data.DataLoader(ds, batch_size=30, shuffle=False)

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.gru1 = nn.GRU(input_size=feature_num, hidden_size=24, batch_first=False, bidirectional=False)
        # self.gru2 = nn.GRU(input_size=24, hidden_size=24, batch_first=False, bidirectional=False)
        self.linear = nn.Linear(24, 24)

    def forward(self, x):
        # print('x.shape',x.shape) # x.shape torch.Size([30, 24, 1])

        x_, _ = self.gru1(x)
        # print('x_.shape',x_.shape) # x_.shape torch.Size([30, 24, 24])

        # x_,_ = self.gru2(x_)
        # print('x_22.shape', x_.shape) #  torch.Size([30, 24, 24])
        # gathering only the latent end-of-sequence for the linear layer

        x_ = x_[:, -1, :]
        # print('x_33.shape',x_.shape) # x_2.shape torch.Size([30, 24])

        out = self.linear(x_)
        # print('out.shape',out.shape) # out.shape torch.Size([30, 24])
        return out

# 定义网络
net = NN()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)
net.train()

# 网络训练
iteration = 0
epochs = 100
for epoch in range(1,epochs+1):
    since = time.time()
    for i, (datapoints, train_labels) in enumerate(dataloader_train):
        train_pred = net(datapoints)
        optimizer.zero_grad()
        loss = criterion(train_pred, train_labels)
        loss.backward()
        optimizer.step()
        iteration += 1
        if iteration % 65 == 0:
            time_elapsed = time.time() - since
            print("epoch：{}, Iteration: {}, Val-loss: {:.4f}, time：{:.0f}s".format(epoch,(iteration),loss,time_elapsed % 60))

# 网络测试
test_pred = net(X_test)
test_pred = test_pred.view(-1,1).data.numpy() # 和reshape功能相同
y_test = y_test.data.numpy()
test_pred = scaler_2.inverse_transform(test_pred)
y_test = scaler_2.inverse_transform(y_test)

data4 = np.array(test_pred).reshape(-1, 1)
data5 = np.array(y_test).reshape(-1, 1)
RMSE = math.sqrt(mean_squared_error(test_pred.reshape(-1, 1), y_test.reshape(-1, 1)))  # 均方误差
MSE = mean_squared_error(test_pred.reshape(-1, 1), y_test.reshape(-1, 1))
MAE = mean_absolute_error(test_pred.reshape(-1, 1), y_test.reshape(-1, 1))
SMAPE = 2.0 * np.mean(np.abs(y_test.reshape(-1, 1) - test_pred.reshape(-1, 1)) / (np.abs(y_test.reshape(-1, 1)) + np.abs(test_pred.reshape(-1, 1)))) * 100
R = np.corrcoef(data4.T, data5.T)
print('RMSE', RMSE, 'MAE', MAE, 'SMAPE', SMAPE, 'MSE', MSE, 'R', R)

plt.figure(figsize=(12, 4))
plt.plot(y_test.reshape(-1,1)[2000:3000])
plt.plot(test_pred.reshape(-1,1)[2000:3000])
plt.legend(['GRU_predict', 'GRU_real'], loc='best')
plt.show()


def save_result(y_test, x_mean):
    y_test = pd.DataFrame(data=y_test.reshape(-1, 1))
    y_test.to_csv(r"..\code\VAE-LSTM-Planar\result\GRU_y_test.csv", index=False, header=['y_test'])
    x_mean = pd.DataFrame(data=x_mean.reshape(-1, 1))
    x_mean.to_csv(r"..\code\VAE-LSTM-Planar\result\GRU_test_pred.csv", index=False, header=['pre_feature'])
# save_result(y_test, test_pred)