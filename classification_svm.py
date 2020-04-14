# %%
"""
# APS1052 Final Project SVM
"""

# %%
"""
# 1. DATA CREATION 
"""

# %%
"""
## 1.1 Importing Libraries
"""

# %%
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
# from fwTimeSeriesSplit import fwTimeSeriesSplit
import pandas as pd
import numpy as np
import talib as ta
import seaborn
import matplotlib.pyplot as plt

from sklearn.metrics import SCORERS
from sklearn.metrics import confusion_matrix
# from sklearn.metrics import plot_confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report



# %%
"""
## 1.2 Importing Data

Upload the data and save it in a Dataframe named as "Df". Use "SPY_adjusted_kibot.txt". Our data is a 30 minute (half hour) OHLCV data for the SPY. Traded from 4AM to 7:30PM (practically round the clock).

Change the Time column to a "datetime" type. And Choose dates between Jan 2009 and Jan 2015 (a subset of it is to be trained).
"""

# %%
Df = pd.read_csv('./data/STOCK/SPY_adjusted_kibot.txt',header = None,dtype={'Date':np.str})
Df.columns=['Date','Time','Open','High','Low','Close','Volume']
Df['Time_hhmm']=Df['Time']
Df['Time']=pd.to_datetime(Df['Date'] + ' ' + Df['Time'])

Start_Point="2008-01-01 03:45:00"
End_Point="2016-01-02 04:15:00"

Df=Df.drop(Df[Df['Time']<Start_Point].index)
Df=Df.drop(Df[Df['Time']>End_Point].index)
##Get a sample dataset for testing
#Df_sample=Df.drop(  Df[Df.index>(Df[Df['Time']>Starting_Point].index)[0]+800].index  )
Df

# %%
"""
## 1.3 Data Visualization (plot k-chart)

### 1.3.1 Plot 2008-2015 Stock Price k-chart (Weekly)
"""

# %%
import plotly.graph_objects as go
#------------------------------------------------------------------------------------------#
# Set Start and End Point for the Plot
Plot_Start_Point="2008-01-01 03:45:00"
Plot_End_Point="2016-01-02 04:15:00"
Df_plot=Df.drop(Df[Df['Time']<Plot_Start_Point].index)
Df_plot=Df_plot.drop(Df[Df['Time']>Plot_End_Point].index)
#------------------------------------------------------------------------------------------#
# Resample the dataframe to get weekly, monthly, Quarterly Stock Price.
stock_data = Df_plot.copy()
period_type = 'W'
stock_data["Date"]
stock_data.set_index('Time',inplace=True)
logic = {'Open'  : 'first',
         'High'  : 'max',
         'Low'   : 'min',
         'Close' : 'last',
         'Volume': 'sum',
         'Date'  : 'first'}
offset = pd.offsets.timedelta(days=-6)
stock_data_re=stock_data.resample(period_type, loffset=offset).apply(logic)
stock_data_re=stock_data_re.reset_index()
#print(stock_data_re.head(20))
#------------------------------------------------------------------------------------------#
# Plot
# Df_plot=stock_data_re
# fig=go.Figure(data=[go.Candlestick(x=Df_plot['Time'],open=Df_plot['Open'], high=Df_plot['High'],low=Df_plot['Low'], close=Df_plot['Close'])])
# #fig.update_layout(xaxis_rangeslider_visible=True)
# fig.update_layout(
#     title='2008-2015 Stock Price k-Chart (Weekly)',
#     yaxis_title='Stock',
#     shapes=[dict(x0='2009-01-01', x1='2009-01-01', y0=0, y1=1, xref='x', yref='paper',line_width=1,line_dash="dot",),
#              dict(x0='2015-01-01', x1='2015-01-01', y0=0, y1=1, xref='x', yref='paper',line_width=1,line_dash="dot",)],
#     annotations=[dict(x='2009-01-01', y=0.95, xref='x', yref='paper',showarrow=False, xanchor='left', text='Training Dataset (2009-2015)')]
# )
# fig.show()

# %%
"""
### 1.3.2 Plot 2008-2015 Stock Price k-chart (Monthly)
"""

# %%
#------------------------------------------------------------------------------------------#
# Set Start and End Point for the Plot
Plot_Start_Point="2008-01-01 03:45:00"
Plot_End_Point="2016-01-02 04:15:00"
Df_plot=Df.drop(Df[Df['Time']<Plot_Start_Point].index)
Df_plot=Df_plot.drop(Df[Df['Time']>Plot_End_Point].index)
#------------------------------------------------------------------------------------------#
# Resample the dataframe to get weekly, monthly, Quarterly Stock Price.
stock_data = Df_plot.copy()
period_type = 'M'
stock_data["Date"]
stock_data.set_index('Time',inplace=True)
logic = {'Open'  : 'first',
         'High'  : 'max',
         'Low'   : 'min',
         'Close' : 'last',
         'Volume': 'sum',
         'Date'  : 'first'}
offset = pd.offsets.timedelta(days=0)
stock_data_re=stock_data.resample(period_type, loffset=offset).apply(logic)
stock_data_re=stock_data_re.reset_index()
#print(stock_data_re.head(20)) # str(stock_data_re['Time'][0])
#------------------------------------------------------------------------------------------#
# Plot
# Df_plot=stock_data_re
# fig=go.Figure(data=[go.Candlestick(x=Df_plot['Time'],open=Df_plot['Open'], high=Df_plot['High'],low=Df_plot['Low'], close=Df_plot['Close'])])
# #fig.update_layout(xaxis_rangeslider_visible=True)
# fig.update_layout(
#     title='2008-2015 Stock Price k-Chart (Monthly)',
#     yaxis_title='Stock',
#     shapes=[dict(x0='2009-01-01', x1='2009-01-01', y0=0, y1=1, xref='x', yref='paper',line_width=1,line_dash="dot",),
#              dict(x0='2015-01-01', x1='2015-01-01', y0=0, y1=1, xref='x', yref='paper',line_width=1,line_dash="dot",)],
#     annotations=[dict(x='2009-01-01', y=0.95, xref='x', yref='paper',showarrow=False, xanchor='left', text='Training Dataset (2009-2015)')]
# )
# fig.show()


# %%
"""
### 1.3.3 Plot 2008-2015 Stock Price k-chart (Quarterly)
"""

# %%
#------------------------------------------------------------------------------------------#
# Set Start and End Point for the Plot
Plot_Start_Point="2008-01-01 03:45:00"
Plot_End_Point="2016-01-02 04:15:00"
Df_plot=Df.drop(Df[Df['Time']<Plot_Start_Point].index)
Df_plot=Df_plot.drop(Df[Df['Time']>Plot_End_Point].index)
#------------------------------------------------------------------------------------------#
# Resample the dataframe to get weekly, monthly, Quarterly Stock Price.
stock_data = Df_plot.copy()
period_type = 'Q'
stock_data["Date"]
stock_data.set_index('Time',inplace=True)
logic = {'Open'  : 'first',
         'High'  : 'max',
         'Low'   : 'min',
         'Close' : 'last',
         'Volume': 'sum',
         'Date'  : 'first'}
offset = pd.offsets.timedelta(days=0)
stock_data_re=stock_data.resample(period_type, loffset=offset).apply(logic)
stock_data_re=stock_data_re.reset_index()
#print(stock_data_re.head(20)) # str(stock_data_re['Time'][0])
#------------------------------------------------------------------------------------------#
# Plot
# Df_plot=stock_data_re
# fig=go.Figure(data=[go.Candlestick(x=Df_plot['Time'],open=Df_plot['Open'], high=Df_plot['High'],low=Df_plot['Low'], close=Df_plot['Close'])])
# #fig.update_layout(xaxis_rangeslider_visible=True)
# fig.update_layout(
#     title='2008-2015 Stock Price k-Chart (Quarterly)',
#     yaxis_title='Stock',
#     shapes=[dict(x0='2009-01-01', x1='2009-01-01', y0=0, y1=1, xref='x', yref='paper',line_width=1,line_dash="dot",),
#              dict(x0='2015-01-01', x1='2015-01-01', y0=0, y1=1, xref='x', yref='paper',line_width=1,line_dash="dot",)],
#     annotations=[dict(x='2009-01-01', y=0.95, xref='x', yref='paper',showarrow=False, xanchor='left', text='Training Dataset (2009-2015)')]
# )
# fig.show()

# %%
"""
### 1.4 Use acf and pacf Plot to Determine Window for Return Periods
"""

# %%
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
Start_Point_2="2009-01-02 00:00:00"
#Get a sample dataset for testing
# Df_sample=Df.drop(Df[Df['Time']<Start_Point_2].index)
# Df_sample=Df_sample.drop(  Df_sample[Df_sample.index>(Df_sample[Df_sample['Time']>Start_Point_2].index)[0]+800].index  )
# plot_acf(Df_sample['Close'], lags=50)
# plot_pacf(Df_sample['Close'], lags=20)
# plt.show()

# %%
n=2 #window for return periods

# %%
"""
# 2. Data Cleaning

## 2.1 Select the range of days to be analyzed.
"""

# %%
#------------------------------------------------------------------------------------------#
# Get the range
Dataset_Start_Point="2008-01-01 03:45:00"
Dataset_End_Point="2016-01-02 04:15:00"
Df_all=Df.drop(Df[Df['Time']<Plot_Start_Point].index)
Df_all=Df_all.drop(Df[Df['Time']>Plot_End_Point].index)
#------------------------------------------------------------------------------------------#
#print(Df_all.head())
Training_Start_Point="2008-01-05 03:45:00"
Df_stock_30m_filled = Df_all.copy()
Df_stock_30m_filled=Df_stock_30m_filled.drop(Df_stock_30m_filled[Df_stock_30m_filled['Time']<Training_Start_Point].index)
#print(Df_stock_30m_filled.head(260))
#print(Df_stock_30m_filled.tail(160))
print(Df_stock_30m_filled.shape)


# %%
"""
## 2.2 Resample the dataframe to fill some gaps in 30 minutes data (missing data points).
"""

# %%
#------------------------------------------------------------------------------------------#
# Resample the dataframe to fill 30 minutes data with Nan's.
period_type = '30t' 
Df_stock_30m_filled.set_index('Time',inplace=True)
logic = {'Open'  : 'first',
         'High'  : 'max',
         'Low'   : 'min',
         'Close' : 'last',
         'Volume': 'sum',
         'Date'  : 'first',
         'Time_hhmm' : 'first'}
offset = pd.offsets.timedelta(days=0)
Df_stock_30m_filled=Df_stock_30m_filled.resample(period_type, loffset=offset).apply(logic)                                 
#print(Df_stock_30m_filled.head(260))
#print(Df_stock_30m_filled.tail(210))
print(Df_stock_30m_filled.shape)

# %%
Df_stock_30m_filled=Df_stock_30m_filled.reset_index()
#Df_stock_30m_filled.fillna(0, inplace=True)

#print(Df_stock_30m_filled.head(260))
#print(Df_stock_30m_filled.tail(210))
print(Df_stock_30m_filled.shape)

Df_stock_30m_filled['Date']= Df_stock_30m_filled.apply (lambda row: str(row['Time'])[0:10], axis=1)
Df_stock_30m_filled['Time_hhmmss']= Df_stock_30m_filled.apply (lambda row: str(row['Time'])[11:], axis=1)
nan_index = Df_stock_30m_filled[  Df_stock_30m_filled.groupby(['Date'])['Open'].transform(max).isna()  ].index
Df_stock_30m_filled=Df_stock_30m_filled.drop(   nan_index   )
Df_stock_30m_filled=Df_stock_30m_filled.reset_index()
#print(Df_stock_30m_filled.head(260))
#print(Df_stock_30m_filled.tail(30))
print(Df_stock_30m_filled.shape)


# %%
Df_stock_30m_filled.fillna(method='ffill', inplace=True)
#(Df_stock_30m_filled == 0).astype(int).sum(axis=0)
Df_stock_30m_filled['Date'].nunique()

# %%
"""
## 2.3 Remove those filled data points that are useless (before 4:00 or after 7:30).
"""

# %%
screening_time_stamp=[]
for i in ["04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19"]:
    for j in ["00","30"]:
        screening_time_stamp.append(i+":"+j+":00")

Df_stock_30m_filled = Df_stock_30m_filled[Df_stock_30m_filled['Time_hhmmss'].isin(screening_time_stamp)]
#print(Df_stock_30m_filled.head(40))
#print(Df_stock_30m_filled.tail())
Df_cleaned=Df_stock_30m_filled.copy()
Df_cleaned.shape # 56352 = 32*1761 rows. We have successfully filled 125 data points.

# %%
"""
## 2.4 Creating indicators
"""

# %%
"""
Use talib package to compute indicators and add to the dataframe.
"""

# %%
Df_cleaned=Df_cleaned.reset_index(drop=True)

Df_cleaned['RSI']=ta.RSI(np.array(Df_cleaned['Close'].shift(1)), timeperiod=n)
Df_cleaned['SMA'] = Df_cleaned['Close'].shift(1).rolling(window=n).mean()
Df_cleaned['Corr']=Df_cleaned['Close'].shift(1).rolling(window=n).corr(Df_cleaned['SMA'].shift(1))
Df_cleaned['SAR']=ta.SAR(np.array(Df_cleaned['High'].shift(1)),np.array(Df_cleaned['Low'].shift(1)),\
                  0.2,0.2)
Df_cleaned['ADX']=ta.ADX(np.array(Df_cleaned['High'].shift(1)),np.array(Df_cleaned['Low'].shift(1)),\
                  np.array(Df_cleaned['Open']), timeperiod =n)

#Df_cleaned['MACD'],_ ,_ = ta.MACD(Df_cleaned['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
#Df_cleaned['KDJ_K'], Df_cleaned['KDJ_D'] = ta.STOCHF(Df_cleaned['High'], Df_cleaned['Low'], Df_cleaned['Close'], fastk_period=9, fastd_period=3, fastd_matype=3)
#Df_cleaned['KDJ_J']= 3 * Df_cleaned['KDJ_D'] - 2 * Df_cleaned['KDJ_K']
#Df_cleaned['EMA12'] = ta.EMA(Df_cleaned['Close'], timeperiod=6)
#Df_cleaned['EMA26'] = ta.EMA(Df_cleaned['Close'], timeperiod=12)
#Df_cleaned['BBOL_u'], Df_cleaned['BBOL_m'], Df_cleaned['BBOL_l'] = ta.BBANDS(Df_cleaned['Close'], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)

print(Df_cleaned.shape)

# %%
Df_cleaned['close'] = Df_cleaned['Close'].shift(1)
Df_cleaned['high'] = Df_cleaned['High'].shift(1)
Df_cleaned['low'] = Df_cleaned['Low'].shift(1)
Df_cleaned['OO']= Df_cleaned['Open']-Df_cleaned['Open'].shift(1)
Df_cleaned['OC']= Df_cleaned['Open']-Df_cleaned['close']

# %%
"""
## 2.5 Add "return" values
Calculate the returns for every data point and save in "Ret". 
"""

# %%
Df_cleaned['Ret']=np.log(Df_cleaned['Open'].shift(-1)/Df_cleaned['Open'])
for i in range(1,n):
    Df_cleaned['return%i'%i]=Df_cleaned['Ret'].shift(i)
#print(Df_cleaned.head(30))
print(Df_cleaned.shape)

# %%
"""
## 2.6 Further trimming the data
"""

# %%
#------------------------------------------------------------------------------------------#
# Get the range
StudySet_Start_Point="2009-01-05 03:45:00"
StudySet_End_Point="2015-01-02 03:45:00"
Df_trimmed=Df_cleaned.drop(Df_cleaned[Df_cleaned['Time']<StudySet_Start_Point].index)
Df_trimmed=Df_trimmed.drop(Df_trimmed[Df_trimmed['Time']>StudySet_End_Point].index)
# generate the Df['Corr'] column
#print(Df_trimmed.tail(40))
#print(Df_trimmed.head(40))
Df_trimmed.to_csv(r'data_temp_1.csv', index = False)
print(Df_trimmed['Date'].nunique())
print(Df_trimmed.shape)
Df_trimmed.loc[Df_trimmed['Corr']<-1,'Corr']=-1
Df_trimmed.loc[Df_trimmed['Corr']>1,'Corr']=1
Df_trimmed.fillna(method='ffill', inplace=True) # fill another 3 nan's
print(Df_trimmed.shape)
#Df_trimmed



# %%
"""
### Creating Output Signals
Assign signal values: "1" is to "Buy","0" is to "Do nothing" and "-1" is to "Sell". 
"""

# %%
#t=.1
#split = int(t*len(Df_trimmed))

#signal_split=123*32 # six month
signal_split=81*32 # four month
#signal_split=81*32 # four month

Df_trimmed['Signal']=0
Df_trimmed.loc[Df_trimmed['Ret']>Df_trimmed['Ret'][:signal_split].quantile(q=0.80),'Signal']=1
Df_trimmed.loc[Df_trimmed['Ret']<Df_trimmed['Ret'][:signal_split].quantile(q=0.20),'Signal']=-1
Df_trimmed.groupby("Signal").count() 


# %%
Df_trimmed.index = np.arange(0, len(Df_trimmed))
#Df_trimmed=Df_trimmed.drop(['index'],axis=1)
Df_trimmed.shape

# %%
"""
# TEST-TEST-TEST-TEST-TEST-TEST-TEST-TEST
# TEST-TEST-TEST-TEST-TEST-TEST-TEST-TEST
"""

# %%
X=Df_trimmed.drop(['Date','Close','Signal','Time_hhmm','Time_hhmmss','High','Low','Volume','Ret','Time'],axis=1)
X2 = X.reset_index()
Y=Df_trimmed['Signal']

X.head()

# %%
steps = [
         		('scaler',StandardScaler()),
        		('svc',SVC())
              ]
pipeline =Pipeline(steps)

tscv = TimeSeriesSplit(n_splits=7)
# tscv_fw = fwTimeSeriesSplit(n_splits=7)

#for grid search
c_gs =[10,100,1000,10000]
g_gs = [1e-2,1e-1,1e0]

#for random search
c_rs = np.linspace(10, 10000, num=40, endpoint=True)
g_rs = np.linspace(1e-2, 1e0, num=30, endpoint=True)

#set of parameters for grid search
parameters_gs = {
              		'svc__C':c_gs,
              		'svc__gamma':g_gs,
              		'svc__kernel': ['rbf', 'poly']
             	           }
#set of parameters for random search
parameters_rs = {
              		'svc__C':c_rs,
              		'svc__gamma':g_rs,
              		'svc__kernel': ['rbf', 'poly']
             	           }

from sklearn.metrics import SCORERS

#cvo = GridSearchCV(pipeline, parameters_gs,cv=7, scoring=None)
#cvo = RandomizedSearchCV(pipeline, parameters_rs,cv=7, scoring=None, n_iter=50, random_state=None)
cvo = RandomizedSearchCV(pipeline, parameters_rs,cv=tscv, scoring=None, n_iter= 50, random_state=70)
#cvo = RandomizedSearchCV(pipeline, parameters_rs,cv=tscv_fw, scoring=None, n_iter=50, random_state=None)
#cvo = RandomizedSearchCV(pipeline, parameters_rs, cv=3, scoring=None, n_iter= 50, random_state=70)

# %%
import time
begin=time.time()
#cvo.fit(X.iloc[:signal_split],Y.iloc[:signal_split])
#best_C = cvo.best_params_['svc__C']
#best_kernel =cvo.best_params_['svc__kernel']
#best_gamma=cvo.best_params_['svc__gamma']
print(time.time()-begin)


# %%
#CV3
best_C =  8463.076923076922
best_kernel =  "rbf"
best_gamma =  0.863448275862069
#solution
best_C = 266.15384615384613
best_kernel = 'poly'
best_gamma = 0.9658620689655173
#_tw
best_C =  3852.307692307692
best_kernel =  "rbf"
best_gamma =  0.5220689655172415

# generate SVM inputs X and Y with time
X_new = Df_trimmed.drop(['Date','Close','Time_hhmm',"Time_hhmmss",'High','Low','Volume','Ret'],axis=1)
X_new = X_new.set_index(X_new['Time'])
X_new = X_new.drop_duplicates(keep='first')
Y_new = X_new['Signal']
print(Y_new)
X_new = X_new.drop(['Signal'],axis=1)
X_new

# for plotting reference
Df_timeasindex = Df_trimmed.set_index(X_new['Time'])
Df_timeasindex = Df_timeasindex.drop_duplicates(keep='first')

class SVMModel(object):
    def __init__(self):
        self.df_result = pd.DataFrame(columns=['Actual', 'Predicted'])

    def get_model(self):
        return SVC(C =best_C,kernel=best_kernel, gamma=best_gamma) 

    def learn(self, df, ys, start_date, end_date, lookback_period):
        model = self.get_model()
        for i in range(0,len(df[start_date:end_date].index),lookback_period):

            date = (df[start_date:end_date].index)[i]
            # Fit the model
            x = self.get_prices_since(df, date, lookback_period*10)
            y = self.get_prices_since(ys, date, lookback_period*10)
            ss1= StandardScaler() 
            model.fit(ss1.fit_transform(x),y)

            # Predict the current period
            index_predict = df.index.get_loc(date)#+lookback_period
            x_current = df.iloc[index_predict:index_predict+lookback_period]          
            y_pred = model.predict(ss1.transform(x_current))

            # Store predictions
            new_index = pd.to_datetime(date, format='%Y-%m-%d')
            y_actual = ys.loc[date]
            self.df_result.loc[new_index] = [y_actual, y_pred]

    def get_prices_since(self, df, date_since, lookback):
        index = df.index.get_loc(date_since)
        return df.iloc[index-lookback:index]
    
test = X_new.copy()
test = test.drop('Time',1)

#Lets define the period which we will use in the model training and result comparision
START = '2009-05'
#END = '2013-01'
END = '2011-01' # do not change

svm_reg_model = SVMModel()
svm_reg_model.learn(test, Y_new, start_date=START, end_date=END, lookback_period=32)

frequency = 2500  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second

# %%

svm_reg_model.df_result.plot(
    title='JPM prediction by OLS', 
    style=['-', '--'], figsize=(12,8))

# %%
Y_pred = svm_reg_model.df_result
# note that in this code we store the prediction of day i+1 on the row of day i.
# therefore we need to do some re-organization of data by trimming off 1 day.
empty_list = []
for i in Y_pred['Predicted']:
  empty_list.extend(i)
Y_pred = np.array(empty_list)

# remove the last day which we do not have X for it to compare
Y_pred = Y_pred[0:len(Y_pred)-32]
len(Y_pred)

# %%
# remove the first day which we did not predict
X_tested = test[START:END].iloc[32:len(test[START:END])]
Df_post = Df_timeasindex[START:END].iloc[32:len(Df_timeasindex)]
Df_post['Pred_Signal']=Y_pred
len(Df_post)
#
# # %%
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import plot_confusion_matrix
# from sklearn.utils.multiclass import unique_labels
# from sklearn.dummy import DummyClassifier
# from sklearn.metrics import classification_report
#
# # %%
# ss1= StandardScaler()
# cls_d = DummyClassifier(strategy='uniform') #can substitute for some other strategy
# cls_d.fit(ss1.fit_transform(test['2009-02':'2009-03']),Y_new['2009-02':'2009-03'])
# y_predict_d =cls_d.predict(ss1.transform(X.iloc[split:]))
#
# #Plot classification report onfusion matrix of dummy
# print(classification_report(Df['Signal'].iloc[split:], y_predict_d))
#
# # %%
# #ploy classification report and confusion matrix of our classifier
# #we also need to remove the last day here in the input
# print(classification_report(Y_new[START:END].iloc[0:len(Y_new[START:END])-17], Y_pred))
# #print(classification_report(Y_new[START:END], Y_pred))
#
# # %%
# Df_post['Ret1']=Df_post['Ret']*Df_post['Pred_Signal']
# Df_post['Cu_Ret1']=0.
#
# # %%
# Df_post['Cu_Ret1']=np.cumsum(Df_post['Ret1'])
#
# # %%
# Df_post['Cu_Ret']=0.
# Df_post['Cu_Ret']=np.cumsum(Df_post['Ret'])
#
# # %%
# Std =np.std(Df_post['Cu_Ret1'])
# Sharpe = (Df_post['Cu_Ret1'].iloc[-1]-Df_post['Cu_Ret'].iloc[-1])/Std #will not annualize this because the data is intraday data
# print('Sharpe Ratio:',Sharpe)
#
# # %%
# plt.plot(Df_post['Cu_Ret1'],color='r',label='Strategy Returns')
# plt.plot(Df_post['Cu_Ret'],color='g',label='Market Returns')
# plt.figtext(0.14,0.7,s='Sharpe ratio: %.2f'%Sharpe)
# plt.legend(loc='best')
# plt.show()
#
# # %%
# #Detrend prices before calculating detrended returns
# Df_post['DetOpen'] = detrendPrice.detrendPrice(Df_post['Open']).values
# #these are the detrended returns to be fed to White's Reality Check
# Df_post['DetRet']=np.log(Df_post['DetOpen'].shift(-1)/Df_post['DetOpen'])
# Df_post['DetStrategy']=Df_post['DetRet']*Df_post['Pred_Signal']
# WhiteRealityCheckFor1.bootstrap(Df_post['DetStrategy'])
#
# # %%
#
#
# # %%
#
#
# # %%
#
#
# # %%
# Y_pred = svm_reg_model.df_result
#
# # note that in this code we store the prediction of day i+1 on the row of day i.
# # therefore we need to do some re-organization of data by trimming off 1 day.
#
# empty_list = []
# for i in Y_pred['Predicted']:
#   empty_list.extend(i)
#
# Y_pred = np.array(empty_list)
#
# # remove the last day which we do not have X for it to compare
# Y_pred = Y_pred[0:len(Y_pred)-17]
#
# # %%
#
# Training_Start_Point="2009-01-05 04:15:00"
# Df_sample=Df.drop(Df[Df['Time']<Training_Start_Point].index)
# Df_sample=Df_sample.drop(  Df[ Df.index> (Df[Df['Time']>Training_Start_Point].index)[0] + 799 ].index  )
#
# Df_sample['Signal']=0
# Df_sample.loc[Df_sample['Ret']>Df_sample['Ret'][:split].quantile(q=0.66),'Signal']=1
# Df_sample.loc[Df_sample['Ret']<Df_sample['Ret'][:split].quantile(q=0.34),'Signal']=-1
# #Df.head()
# X=Df_sample.drop(['Date','Close','Signal','High','Low','Volume','Ret'],axis=1)
# y=Df_sample['Signal']
# X.head()
#
# # %%
# Df_sample.groupby("Signal").count() #shows -1, 0 and 1 occur in roughly equal numbers
#
# # %%
# """
# ### Data Visualization (for the very first training dataset)
# 1. Plot for ['SMA','RSI']
# """
#
# # %%
# X2D = X[['SMA','RSI']]
# print("Our selected features: \n", X2D.head(1))
# X2D_arr=X2D.values
# print("Our X2D, first three rows: \n", X2D_arr[0:3,:])
# y_arr=y.values
# print("Our y values, first six rows: ", y_arr[0:6])
#
# # %%
# # plot
# plt.figure(figsize=(5, 5))
# plt.scatter(X2D_arr[:, 0], X2D_arr[:, 1], c=y_arr)
# plt.legend(["Class 0","Class 1","Class 1"], loc=2)
# plt.xlabel("First feature")
# plt.ylabel("Second feature")
# print("X2D.shape:", X2D_arr.shape)
#
# # %%
# """
# 2. Plot for ['ADX','Corr'].
# """
#
# # %%
# X2D = X[['ADX','Corr']]
# print("Our selected features: \n", X2D.head(1))
# X2D_arr=X2D.values
# print("Our X2D, first three rows: \n", X2D_arr[0:3,:])
# y_arr=y.values
# print("Our y values, first six rows: ", y_arr[0:6])
#
# # %%
# # plot
# plt.figure(figsize=(5, 5))
# plt.scatter(X2D_arr[:, 0], X2D_arr[:, 1], c=y_arr)
# plt.legend(["Class 0","Class 1","Class 1"], loc=2)
# plt.xlabel("First feature")
# plt.ylabel("Second feature")
# print("X2D.shape:", X2D_arr.shape)
#
# # %%
# """
# 3. Plot for ['SAR','ADX'].
# """
#
# # %%
# X2D = X[['SAR','ADX']]
# print("Our selected features: \n", X2D.head(1))
# X2D_arr=X2D.values
# print("Our X2D, first three rows: \n", X2D_arr[0:3,:])
# y_arr=y.values
# print("Our y values, first six rows: ", y_arr[0:6])
#
# # %%
# # plot
# plt.figure(figsize=(5, 5))
# plt.scatter(X2D_arr[:, 0], X2D_arr[:, 1], c=y_arr)
# plt.legend(["Class 0","Class 1","Class 1"], loc=2)
# plt.xlabel("First feature")
# plt.ylabel("Second feature")
# print("X2D.shape:", X2D_arr.shape)
#
# # %%
# """
# # >>> Using "Harness" to train SVM weekly (or daily)
# """
#
# # %%
# """
#
# """
#
# # %%
#
#
# # %%
# """
# ## Train the model based on randomized search
# """
#
# # %%
#
# Plot_Start_Point="2008-01-01 04:15:00"
# Plot_End_Point="2016-01-01 04:15:00"
# Df_plot=Df.drop(Df[Df['Time']<Plot_Start_Point].index)
# Df_plot=Df_plot.drop(Df[Df['Time']>Plot_End_Point].index)
# #------------------------------------------------------------------------------------------#
# #------------------------------------------------------------------------------------------#
# print(Df_plot.head(20))
#
# Training_Start_Point="2009-01-05 04:15:00"
#
# # Resample the dataframe to get weekly, monthly, Quarterly Stock Price.
#
# stock_weekly_data = Df_plot.copy()
# period_type = '30t'
# stock_weekly_data.set_index('Time',inplace=True)
# logic = {'Open'  : 'first',
#          'High'  : 'max',
#          'Low'   : 'min',
#          'Close' : 'last',
#          'Volume': 'sum',
#          'Date'  : 'first'}
# offset = pd.offsets.timedelta(days=0)
# stock_weekly_data=stock_weekly_data.resample(period_type, loffset=offset).apply(logic)
# stock_weekly_data=stock_weekly_data.reset_index()
# print(stock_weekly_data.head(20))
#
# Training_Start_list=[]
#
# Df_sample=Df.drop(Df[Df['Time']<Training_Start_Point].index)
# Df_sample=Df_sample.drop(  Df[ Df.index> (Df[Df['Time']>Training_Start_Point].index)[0] + 1599 ].index  )
# t=.8
# split = int(t*len(Df_sample))
#
#
#
#
#
#
#
#
#
# # # %%
# # time_begin=time.time()
# # from SVC_model import *
# # Sharpe, WhiteRealityCheck_p_value, Df_current_trained = SVC_model(Df_sample, n=10, cv_model="tscv",random_state_model=1)
# # print (Sharpe, WhiteRealityCheck_p_value)
# # print (time.time()-time_begin)