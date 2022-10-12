# PFVAE-A-Planar-Flow-Based-Variational-Auto-Encoder-Prediction-Model-for-Time-Series-Data
## Abstract
Prediction based on time series has a wide range of applications. Due to the complex nonlinear and random distribution of time series data, the performance of learning prediction models can be reduced by the modeling bias or overfitting. This paper proposes a novel planar flow-based variational auto-encoder prediction model (PFVAE), which uses the long- and short-term memory network (LSTM) as the auto-encoder and designs the variational auto-encoder (VAE) as a time series data predictor to overcome the noise effects. In addition, the internal structure of VAE is transformed using planar flow, which enables it to learn and fit the nonlinearity of time series data and improve the dynamic adaptability of the network. The prediction experiments verify that the proposed model is superior to other models regarding prediction accuracy and proves it is effective for predicting time series data.
## Author
Jin, Xue-Bo, Wen-Tao Gong, Jian-Lei Kong, Yu-Ting Bai, and Ting-Li Su.
## Cite this paper
Jin, Xue-Bo, Wen-Tao Gong, Jian-Lei Kong, Yu-Ting Bai, and Ting-Li Su. 2022. "PFVAE: A Planar Flow-Based Variational Auto-Encoder Prediction Model for Time Series Data" Mathematics 10, no. 4: 610.
## Link to paper  
https://www.mdpi.com/2227-7390/10/4/610  
## runnning code  
The ``PFVAE.py``is a neural network model based on ``Planar`` combining ``LSTM ``and ``VAE``. PM2.5 data, temperature data and humidity data are input into this program to predict, and the prediction results are obtained and saved locally.    
```
PFVAE.py
```
``LSTM.py``、``GRU.py``、``CNN-LSTM.py``、``Conv-LSTM.py``、``op_ Blstm.py``, ``origin-vae.py``, and ``tcn Py`` is the neural network model used in this study to compare with ``PFVAE`` neural network model, and the prediction results of each model are also saved locally.  
  
``violin_ Fig.py`` is mainly used to draw violin pictures.    
  
``quxiantu. py`` is mainly a program to draw corresponding prediction curves according to the prediction results of each model.  
  
``fenbutu. py`` mainly applies the standardized flow method to complicate the simulated simple Gaussian distribution into a relatively complex distribution.  
## Data set
```
guanyuan.csv
```
