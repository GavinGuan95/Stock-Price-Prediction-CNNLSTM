# stock-price-prediction-with-CNN-LSTM
This repository contains the source code for the paper "Stock Price Prediction with CNN-LSTM Network". <br />
It performs next day stock price prediction using Dilated CNN, or LSTM, or the combination of both. It also performs simulated training based on the next day stock price prediction.
It is fond that a combination of Dilated CNN and LSTM give the most accurate prediciton. We have also compared the Neural Network's performance with classical machine learning methods such as SVM, Random Forest and Linear Regression.  

# Training and Testing
Training
```
./train.py # generate default results
./data_loader/configs # location of generated result
```
Modify Configuration
```
vim ./data_loader/configs/config1.json
vim ./data_loader/configs/config2.json
```

For the non-neural methods, simply call their corresponding files:
```
./KNN.py
./RandomForest.py
./Linear_Regression.py
```
