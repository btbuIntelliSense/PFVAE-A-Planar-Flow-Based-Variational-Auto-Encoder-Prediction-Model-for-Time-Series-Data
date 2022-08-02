import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch
def jubufangda():
    '''
    局部放大折线图画法
    :return:
    '''
    start = 4000
    end = 5000
    MAX_EPISODES = end - start
    x_axis_data = []
    for l in range(MAX_EPISODES):
        x_axis_data.append(l)

    real_path = r'F:\66的研究生日子\科研生活\研二 -- 上\论文2\code\VAE-LSTM-Planar\result\tem\NF-VAE_y_test.csv'
    LSTM_path = r'F:\66的研究生日子\科研生活\研二 -- 上\论文2\code\VAE-LSTM-Planar\result\tem\LSTM_test_pred.csv'
    GRU_path = r'F:\66的研究生日子\科研生活\研二 -- 上\论文2\code\VAE-LSTM-Planar\result\tem\GRU_test_pred.csv'
    TCN_path = r'F:\66的研究生日子\科研生活\研二 -- 上\论文2\code\VAE-LSTM-Planar\result\tem\TCN_preds_test.csv'
    CNNLSTM_path = r'F:\66的研究生日子\科研生活\研二 -- 上\论文2\code\VAE-LSTM-Planar\result\tem\CNN-LSTM_test_pred.csv'
    ConvLSTM_path = r'F:\66的研究生日子\科研生活\研二 -- 上\论文2\code\VAE-LSTM-Planar\result\tem\Convlstm_test_pred.csv'
    BayesLSTM_path = r'F:\66的研究生日子\科研生活\研二 -- 上\论文2\code\VAE-LSTM-Planar\result\tem\Bayes_preds_test.csv'
    vae_path =  r'F:\66的研究生日子\科研生活\研二 -- 上\论文2\code\VAE-LSTM-Planar\result\tem\VAE_preds_test.csv'
    PVl_path = r'F:\66的研究生日子\科研生活\研二 -- 上\论文2\code\VAE-LSTM-Planar\result\tem\NF-VAE_preds_test.csv'

    real_data = pd.read_csv(real_path).values[start:end].reshape(MAX_EPISODES,1)
    LSTM_data = pd.read_csv(LSTM_path).values[start:end].reshape(MAX_EPISODES,1)
    GRU_data = pd.read_csv(GRU_path).values[start:end].reshape(MAX_EPISODES, 1)
    TCN_data = pd.read_csv(TCN_path).values[start:end].reshape(MAX_EPISODES, 1)
    CNNLSTM_data = pd.read_csv(CNNLSTM_path).values[start:end].reshape(MAX_EPISODES, 1)
    ConvLSTM_data = pd.read_csv(ConvLSTM_path).values[start:end].reshape(MAX_EPISODES, 1)
    BayesLSTM_data = pd.read_csv(BayesLSTM_path).values[start:end].reshape(MAX_EPISODES, 1)
    vae_data = pd.read_csv(vae_path).values[start:end].reshape(MAX_EPISODES, 1)
    PVl__data = pd.read_csv(PVl_path).values[start:end].reshape(MAX_EPISODES, 1)

    caihongse = ["#FF0000,#FF7F00,#FFFF00, #00FF00 , #00FFFF ,#0000FF,#8B00FF "]

    #设置x轴标签和尺寸
    plt.rcParams['axes.labelsize']=16
    plt.rcParams['xtick.labelsize'] = 15  # x轴ticks的size
    plt.rcParams['ytick.labelsize'] = 15  # y轴ticks的size
    plt.rcParams.update({'font.size': 14})

    fig, ax = plt.subplots(1, 1,figsize=(20, 10))
    ax.plot(x_axis_data, LSTM_data, color='#FF7F00', alpha=0.8, label='LSTM')
    ax.plot(x_axis_data, BayesLSTM_data, color='#00FFFF', alpha=0.8, label='BayesLSTM')
    ax.plot(x_axis_data, GRU_data, color='#FFFF00', alpha=0.8, label='GRU')
    ax.plot(x_axis_data, CNNLSTM_data, color='#FF0000', alpha=0.8, label='CNN-LSTM')
    ax.plot(x_axis_data, TCN_data, color='#0000FF', alpha=0.8, label='TCN')
    ax.plot(x_axis_data, ConvLSTM_data, color='#00FF00', alpha=0.8, label='ConvLSTM')
    ax.plot(x_axis_data, vae_data, alpha=0.8, label='VAE')# color='#FF3EFF',
    ax.plot(x_axis_data, PVl__data, color='#8B00FF', alpha=0.8, label='Our proposed model')
    ax.plot(x_axis_data, real_data, color='#000000', alpha=0.8, label='real')
    plt.legend(loc='lower center',ncol=5,bbox_to_anchor=(0.5,-0.14)) # 把图例设置在外边
    # plt.ylabel('Death number')
    # #嵌入局部放大图的坐标系
    # axins = inset_axes(ax, width="40%", height="30%", loc='lower left',
    #                    bbox_to_anchor=(0.53, 0.5, 1.0, 1.2),
    #                    bbox_transform=ax.transAxes)
    # #在子坐标系中绘制原始数据
    # axins.plot(x_axis_data, real_data, color='#7030A0', alpha=0.8)
    # axins.plot(x_axis_data, LSTM_data, color='#940000', alpha=0.8)
    # axins.plot(x_axis_data, GRU_data, color='#CD853F', alpha=0.8)
    # axins.plot(x_axis_data, TCN_data, color='#E7A5AF', alpha=0.8)
    # axins.plot(x_axis_data, CNNLSTM_data, color='#627758', alpha=0.8)
    # axins.plot(x_axis_data, ConvLSTM_data, color='#92D050', alpha=0.8)
    # axins.plot(x_axis_data, BayesLSTM_data, color='#4DC3E8', alpha=0.8)
    # axins.plot(x_axis_data, PVl__data, color='#4DC3E8', alpha=0.8)
    #
    # #设置放大区间，调整子坐标系的显示范围
    # # 设置放大区间
    # zone_left = 50
    # zone_right = 100
    #
    # # 坐标轴的扩展比例（根据实际数据调整）
    # x_ratio = 0.0  # x轴显示范围的扩展比例
    # y_ratio = 0.07  # y轴显示范围的扩展比例
    #
    # # X轴的显示范围
    # xlim0 = x_axis_data[zone_left] - (x_axis_data[zone_right] - x_axis_data[zone_left]) * x_ratio
    # xlim1 = x_axis_data[zone_right] + (x_axis_data[zone_right] - x_axis_data[zone_left]) * x_ratio
    #
    #
    # # Y轴的显示范围
    # y = np.hstack((real_data[zone_left:zone_right], Bayes_data[zone_left:zone_right],
    #                GRU_data[zone_left:zone_right], LSTM_data[zone_left:zone_right],
    #                ConvLSTM_data[zone_left:zone_right],CNNLSTM_data[zone_left:zone_right],
    #                TCN_data[zone_left:zone_right]))
    # ylim0 = np.min(y) - (np.max(y) - np.min(y)) * y_ratio
    # ylim1 = np.max(y) + (np.max(y) - np.min(y)) * y_ratio
    #
    # # 调整子坐标系的显示范围
    # axins.set_xlim(xlim0, xlim1)
    # axins.set_ylim(ylim0, ylim1)
    #
    # #建立父坐标系与子坐标系的连接线
    # # 原图中画方框
    # tx0 = xlim0
    # tx1 = xlim1
    # ty0 = ylim0
    # ty1 = ylim1
    # sx = [tx0, tx1, tx1, tx0, tx0]
    # sy = [ty0, ty0, ty1, ty1, ty0]
    # ax.plot(sx, sy, "#CD853F")
    #
    # print('xlim0 : ',xlim0)
    # print('xlim1 : ', xlim1)
    # print('ylim0 : ', ylim0)
    # print('ylim1 : ', ylim1)
    # # 画两条线
    # xy = (xlim0, ylim0) #(60,-409)
    # xy2 = (xlim0, ylim0)#(60,409)
    # con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
    #                       axesA=axins, axesB=ax)
    # axins.add_artist(con)
    #
    # xy = (xlim1, ylim0)#(90,-409)
    # xy2 = (xlim1, ylim0)#(90,-409)
    # con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
    #                       axesA=axins, axesB=ax)
    # axins.add_artist(con)

    # plt.savefig(r'F:\66的研究生日子\科研生活\研二 -- 上\论文2\picture\pm2.5-quxiantu.png', bbox_inches='tight', dpi=600)  # 高清图
    plt.show()

if __name__=="__main__":
    jubufangda()