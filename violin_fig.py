import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

tips = pd.read_excel(r'F:\66的研究生日子\科研生活\研二 -- 上\论文2\code\VAE-LSTM-Planar\result\PM2.5\error_20.xlsx')
plt.figure(figsize=(20,10))
sns.violinplot(data=tips,
               split=True,
               linewidth = 2, #线宽
               width = 0.9,   #箱之间的间隔比例
               palette = 'muted', #设置调色板
               # scale = 'count',  #测度小提琴图的宽度： area-面积相同,count-按照样本数量决定宽度,width-宽度一样
               gridsize = 50, #设置小提琴图的平滑度，越高越平滑
               # inner = 'box', #设置内部显示类型 --> 'box','quartile','point','stick',None
               #bw = 0.8      #控制拟合程度，一般可以不设置
               )

plt.xticks(ticks=[0,1,2,3,4,5,6,7,8,9], labels=['LSTM' , 'BiLSTM' , 'GRU' , 'BiGRU', 'CNNLSTM',
                                                'ConvLSTM','TCN','BayesLSTM','VAE','PFVAE',],
           fontsize=16, ha='center', va='top')
plt.ylabel('RMSE',fontsize=18)
plt.yticks(fontsize=16)
# plt.legend(labels=['LSTM' , 'BiLSTM' , 'GRU' , 'BiGRU', 'TCN','CNNLSTM','ConvLSTM','BayesLSTM','VAE','Our proposed method'],
#            loc='lower center',ncol=5,bbox_to_anchor=(0.5,-0.14)) # 把图例设置在外边
plt.show()