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
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.svm import SVR
from sklearn import datasets
from sklearn.metrics import accuracy_score
from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import h5py

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

target[target <= 0] = 0
target[target > 0] = 1


#split train test to 7:3
t=.7
split = int(t*len(X))

x_tr=X_new.iloc[:split]
x_te=X_new.iloc[split:]
y_tr=target.iloc[:split]
y_te=target.iloc[split:]

output=np.array(y_tr)
y_tr=output.ravel()
#set of parameters for random search
parameters = {'gamma': [0.00001, 0.0001, 0.01], 'C': [1, 10], 'kernel': ['rbf', 'poly']}
svc_g = SVC()
svc_searched = RandomizedSearchCV(svc_g, parameters, cv = 5, n_iter = 5, verbose=0, n_jobs = -1)

svc_searched.fit(x_tr,y_tr)
print(svc_searched.best_params_)
svc_predict = svc_searched.best_estimator_
print(svc_predict.get_params)
svc_predict.fit(x_tr,y_tr)
svc_pred_tr = svc_predict.predict(x_tr)
#print train and validation accuracy
print("SVM training accuracy for next day", accuracy_score(y_tr, svc_pred_tr))
svc_pred_val = svc_predict.predict(x_te)
print("SVM testing accuracy for next day", accuracy_score(y_te, svc_pred_val))

#print f1 score
print("the f1 score for SVM next day prediction is", f1_score(y_te, svc_pred_val, average='binary'))

#print confusion metrics
print("confusion matrix for next day prediction, class 0 is decrease and class 1 is increase")

print(confusion_matrix(y_te, svc_pred_val))

# code below is for SVM on 10 day average movement
#choose target to be the average movement of next 10 day price
target = Df['FSMA_10']
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

target[target <= 0] = 0
target[target > 0] = 1


#split train test to 7:3
t=.7
split = int(t*len(X))

x_tr=X_new.iloc[:split]
x_te=X_new.iloc[split:]
y_tr=target.iloc[:split]
y_te=target.iloc[split:]

output=np.array(y_tr)
y_tr=output.ravel()


#set of parameters for random search
parameters = {'gamma': [0.00001, 0.0001, 0.01], 'C': [1, 10], 'kernel': ['rbf', 'poly']}
svc_g = SVC()
svc_searched = RandomizedSearchCV(svc_g, parameters, cv = 5, n_iter = 5, verbose=0, n_jobs = -1)

svc_searched.fit(x_tr,y_tr)
print(svc_searched.best_params_)
svc_predict = svc_searched.best_estimator_
print(svc_predict.get_params)
svc_predict.fit(x_tr,y_tr)
svc_pred_tr = svc_predict.predict(x_tr)
#print train and validation accuracy
print("SVM training accuracy for 10 day average",accuracy_score(y_tr, svc_pred_tr))
svc_pred_val = svc_predict.predict(x_te)
print("SVM testing accuracy for 10 day average",accuracy_score(y_te, svc_pred_val))

#print f1 score
print("the f1 score for svc 10 days average is", f1_score(y_te, svc_pred_val, average='binary'))

#print confusion metrics
print("confusion matrix for 10 day average, class 0 is decrease and class 1 is increase")

print(confusion_matrix(y_te, svc_pred_val))