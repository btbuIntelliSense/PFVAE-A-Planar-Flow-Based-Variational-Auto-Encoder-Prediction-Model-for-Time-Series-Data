import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch

def shuangYaxisBar():
    '''
    绘制双Y轴柱状图
    :return:
    '''
    # LSTM1 = np.array([33.45, 24.59, 67.06])
    # BayesLSTM1 = np.array([30.03, 23.16, 69.47])
    # GRU1 = np.array([28.93, 21.37, 62.09])
    # CNNLSTM1 = np.array([28.78, 20.52, 59.94])
    # TCN1 = np.array([26.77, 18.46, 61.41])
    # ConvLSTM1 = np.array([26.36,18.11, 55.74])
    # vae1 = np.array([25.84,17.72, 53.08])
    # PlnaerVAE1 = np.array([24.45, 17.07, 51.29])
    # labels = ['RMSE', 'MAE', 'SMAPE']
    #----------------------------------------------
    GRU1 = np.array([4.22, 3.32, 0])
    vae1 = np.array([3.42, 2.61, 0])
    CNNLSTM1 = np.array([3.35, 2.53, 0])
    LSTM1 = np.array([3.34, 2.48,0])
    ConvLSTM1 = np.array([3.24, 2.53, 0])
    TCN1 = np.array([3.05, 2.20, 0])
    BayesLSTM1 = np.array([3.03,2.30,0])
    PlnaerVAE1 = np.array([2.93,2.20,0])

    GRU2 = np.array([0, 0, 37.00])
    vae2 = np.array([0, 0, 36.74])
    CNNLSTM2 = np.array([0, 0, 35.81])
    LSTM2 = np.array([0,0, 33.31])
    ConvLSTM2 = np.array([0, 0, 33.70])
    TCN2 = np.array([0, 0, 32.05])
    BayesLSTM2 = np.array([0, 0,32.83])
    PlnaerVAE2 = np.array([0,0,31.61])
    labels = ['RMSE', 'MAE', 'SMAPE']

    #设置x轴标签和尺寸
    plt.rcParams['axes.labelsize']=16
    plt.rcParams['xtick.labelsize'] = 15  # x轴ticks的size
    plt.rcParams['ytick.labelsize'] = 15  # y轴ticks的size
    plt.rcParams.update({'font.size': 14})

    #设置柱间的距离
    width = 0.11  # 柱形的宽度
    x1_list = []
    x2_list = []
    x3_list = []
    x4_list = []
    x5_list = []
    x6_list = []
    x7_list = []
    x8_list = []

    for i in range(len(LSTM1)):
        x1_list.append(i)
        x2_list.append(i + width)
        x3_list.append(i + 2 * width)
        x4_list.append(i + 3 * width)
        x5_list.append(i + 4 * width)
        x6_list.append(i + 5 * width)
        x7_list.append(i + 6 * width)
        x8_list.append(i + 7 * width)
    colors1 = ['#2E3F5B', '#CF9CBB', '#5A704C', '#DA9F5D']
    colors1 = ['#B4A8DA', '#E27A93', '#7EE0D1', '#F6C988']
    colors1 = ['#D7B77A', '#416170', '#E7A5AF', '#627758']
    colors1 = ['#6BC7F8', '#E54688', '#7EE0D1', '#EFB88F']

    caihongse = ["#FF0000,#FF7F00,#FFFF00, #00FF00 , #00FFFF ,#0000FF,#8B00FF "]
    #创建图层
    # fig,ax1=plt.subplots(figsize=(18,10))
    # ax1.set_ylim(0,80)
    # b1 = plt.bar(x1_list, LSTM1, width=width, label='LSTM', color='#FF7F00', align='edge')
    # b2 = plt.bar(x2_list, BayesLSTM1, width=width, label='BayesLSTM', color='#00FFFF', align='edge')
    # b3 = plt.bar(x3_list, GRU1, width=width, label='GRU', color='#FFFF00', align='edge')
    # b4 = plt.bar(x4_list, CNNLSTM1, width=width, label='CNN-LSTM', color='#FF0000', align='edge', tick_label=labels)
    # b5 = plt.bar(x5_list, TCN1, width=width, label='TCN', color='#0000FF', align='edge')
    # b6 = plt.bar(x6_list,ConvLSTM1, width=width, label='ConvLSTM', color='#00FF00', align='edge')
    # b7 = plt.bar(x7_list, vae1, width=width, label='VAE', align='edge')#, color='#8B00FF'
    # b8 = plt.bar(x8_list, PlnaerVAE1, width=width, label='Our proposed method', color='#8B00FF', align='edge')
    # -------------------------------------------------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(16, 10))
    ax1.set_ylim(0,5)
    ax1.vlines([1.94], 0,4, linestyles='dashed', colors='black')  # 加虚线
    ax1.bar(x1_list, GRU1, width=width, label='GRU', color='#FFFF00', align='edge')
    plt.bar(x2_list, vae1, width=width, label='VAE', align='edge')  # color='#0000FF',
    plt.bar(x3_list, CNNLSTM1, width=width, label='CNN-LSTM', color='#FF0000', align='edge')
    plt.bar(x4_list, LSTM1, width=width, label='LSTM', color='#FF7F00', align='edge', tick_label=labels)
    plt.bar(x5_list, ConvLSTM1, width=width, label='ConvLSTM', color='#00FF00', align='edge')
    plt.bar(x6_list, TCN1, width=width, label='TCN', color='#0000FF', align='edge')
    plt.bar(x7_list, BayesLSTM1, width=width, label='BayesLSTM', color='#00FFFF', align='edge')
    plt.bar(x8_list, PlnaerVAE1, width=width, label='Our proposed method', color='#8B00FF', align='edge')

    ax2=ax1.twinx()
    ax2.set_ylim(0, 45)
    b1=plt.bar(x1_list, GRU2, width=width, label='GRU', color='#FFFF00', align='edge')
    b2=plt.bar(x2_list, vae2, width=width, label='VAE', align='edge')  # color='#0000FF',
    b3=plt.bar(x3_list, CNNLSTM2, width=width, label='CNN-LSTM', color='#FF0000', align='edge')
    b4=plt.bar(x4_list, LSTM2, width=width, label='LSTM', color='#FF7F00', align='edge', tick_label=labels)
    b5=plt.bar(x5_list, ConvLSTM2, width=width, label='ConvLSTM', color='#00FF00', align='edge')
    b6=plt.bar(x6_list, TCN2, width=width, label='TCN', color='#0000FF', align='edge')
    b7=plt.bar(x7_list, BayesLSTM2, width=width, label='BayesLSTM', color='#00FFFF', align='edge')
    b8=plt.bar(x8_list, PlnaerVAE2, width=width, label='Our proposed method', color='#8B00FF', align='edge')

    plt.legend(handles=[b1,b2,b3,b4,b5,b6,b7,b8],loc='lower center',ncol=4,bbox_to_anchor=(0.5,-0.14))
    plt.savefig(r'F:\66的研究生日子\科研生活\研二 -- 上\论文2\picture\tem-双坐标误差图-rmse.png', bbox_inches='tight', dpi=600)  # 高清图

    # ax = plt.gca()
    # spines是指连接坐标轴的线，一共有上下左右四个。top/bottom/right/left
    # 上面的和右面的设置成无色
    # ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')
    # plt.grid(False)  # 去掉网格线
    plt.show()

if __name__=="__main__":
    shuangYaxisBar()