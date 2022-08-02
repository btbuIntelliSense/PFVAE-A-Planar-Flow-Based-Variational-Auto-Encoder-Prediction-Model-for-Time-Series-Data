# PFVAE-A-Planar-Flow-Based-Variational-Auto-Encoder-Prediction-Model-for-Time-Series-Data
## Link to paper  
https://www.mdpi.com/2227-7390/10/4/610  
## runnning code  
The ``PFVAE.py`` main program under ``VAE LSTM planar file`` is a neural network model based on ``Planar`` combining ``LSTM ``and ``VAE``. PM2.5 data, temperature data and humidity data are input into this program to predict, and the prediction results are obtained and saved locally.    
  
``LSTM.py``、``GRU.py``、``CNN-LSTM.py``、``Conv-LSTM.py``、``op_ Blstm.py``, ``origin-vae.py``, and ``tcn Py`` is the neural network model used in this study to compare with ``PFVAE`` neural network model, and the prediction results of each model are also saved locally.  
  
``violin_ Fig.py`` is mainly used to draw violin pictures.    
  
``quxiantu. py`` is mainly a program to draw corresponding prediction curves according to the prediction results of each model.  
  
``fenbutu. py`` mainly applies the standardized flow method to complicate the simulated simple Gaussian distribution into a relatively complex distribution.  
