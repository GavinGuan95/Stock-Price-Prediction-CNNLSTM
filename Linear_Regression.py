#import all libraries
import talib as ta
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import io
from PIL import Image
from sklearn import datasets
from sklearn.metrics import accuracy_score
from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
import h5py
from sklearn.metrics import f1_score

#load data
Df= pd.read_csv('./data_loader/processed_data/spy_processed.csv')
Df=Df.dropna()

#choose target to be the movement of next day price
target = Df['ROC_1']
X = Df.drop(['ROC_1', 'Date', 'Volume', 'FSMA_10'], axis=1)

# normalize the data
sc_X = StandardScaler()
X= sc_X.fit_transform(X)
X_new = pd.DataFrame(data=X)
# set the context window size
window_size = 10

input_list = []
output_list = []
for idx, output in enumerate(target):
    if idx < window_size:
        continue
    input = X_new.iloc[range(idx-window_size, idx), :].to_numpy().ravel()
    input_list.append(input)
    output_list.append(output)

    dummy = 1

assert(len(input_list) == len(output_list))
X_new=pd.DataFrame(input_list)
target=pd.DataFrame(output_list)
class_target = target

class_target[class_target <= 0] = 0
class_target[class_target > 0] = 1


#split train test to 7:3
t=.7
split = int(t*len(X))

x_tr=X_new.iloc[:split]
x_te=X_new.iloc[split:]
y_tr=target.iloc[:split]
y_te=target.iloc[split:]

class_tr=class_target.iloc[:split]
class_te=class_target.iloc[split:]

output=np.array(y_tr)
y_tr=output.ravel()

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error
lr_predict = LinearRegression().fit(x_tr,y_tr)

lr_pred_tr = lr_predict.predict(x_tr)
#print train and validation accuracy

print("next day training MSE",mean_squared_error(y_tr, lr_pred_tr))
lr_pred_val = lr_predict.predict(x_te)
print("nest day testing MSE",mean_squared_error(y_te, lr_pred_val))

print("next day MAPE", mean_absolute_error(y_te, lr_pred_val))

p_target=pd.DataFrame(lr_pred_val)

p_target[p_target <= 0] = 0
p_target[p_target > 0] = 1

print("next day accuracy",accuracy_score(class_te, p_target))

#print f1 score
print("the f1 score for LR next day", f1_score(class_te, p_target, average='binary'))

#print confusion metrics
print("confusion matrix for 10 day average, class 0 is decrease and class 1 is increase")

print(confusion_matrix(class_te, p_target))


# code below is for LN on 10 day average trend
#load data
Df= pd.read_csv('./data_loader/processed_data/spy_processed.csv')
Df=Df.dropna()

#choose target to be the average movement of next 10 day price
target = Df['FSMA_10']
X = Df.drop(['ROC_1', 'Date', 'Volume', 'FSMA_10'], axis=1)

# normalize the data
sc_X = StandardScaler()
X= sc_X.fit_transform(X)
X_new = pd.DataFrame(data=X)
# set the context window size

# set the context window size
window_size = 10

input_list = []
output_list = []
for idx, output in enumerate(target):
    if idx < window_size:
        continue
    input = X_new.iloc[range(idx-window_size, idx), :].to_numpy().ravel()
    input_list.append(input)
    output_list.append(output)

    dummy = 1

assert(len(input_list) == len(output_list))
X_new=pd.DataFrame(input_list)
target=pd.DataFrame(output_list)
class_target = target

class_target[class_target <= 0] = 0
class_target[class_target > 0] = 1


#split train test to 7:3
t=.7
split = int(t*len(X))

x_tr=X_new.iloc[:split]
x_te=X_new.iloc[split:]
y_tr=target.iloc[:split]
y_te=target.iloc[split:]

class_tr=class_target.iloc[:split]
class_te=class_target.iloc[split:]

output=np.array(y_tr)
y_tr=output.ravel()

from sklearn.metrics import mean_squared_error
lr_predict = LinearRegression().fit(x_tr,y_tr)

lr_pred_tr = lr_predict.predict(x_tr)
#print train and validation accuracy

print("trainign 10 day average MSE", mean_squared_error(y_tr, lr_pred_tr))
lr_pred_val = lr_predict.predict(x_te)
print("testing 10 day average MSE", mean_squared_error(y_te, lr_pred_val))

print("10 day average MAPE", mean_absolute_error(y_te, lr_pred_val))

p_target=pd.DataFrame(lr_pred_val)

p_target[p_target <= 0] = 0
p_target[p_target > 0] = 1

print("10 day average accuracy", accuracy_score(class_te, p_target))

#print f1 score
print("the f1 score for svc 10 days average is", f1_score(class_te, p_target, average='binary'))

#print confusion metrics
print("confusion matrix for 10 day average, class 0 is decrease and class 1 is increase")

print(confusion_matrix(class_te, p_target))