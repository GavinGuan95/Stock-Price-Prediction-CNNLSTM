"""
All non-neural data preprocessing.

Technical indicators:

MA - moving average
EMA - exponential moving average
MOM - momentum
ROC - rate of change
ATR - average true range
BBands - bollinger bands
RSI - relative strength index
MACD - Moving Average Convergence Divergencew
william_r_% - Williams Percent Range
k% - slow stochastic indicator
D% - fast stochastic indicator
A/D - accumulation/distribution


"""
import os
import numpy as np
import pandas as pd
import talib as ta

from pandas import ExcelWriter
from pandas import ExcelFile

# TODO: derive adjusted Open, High, Low from the downloaded data. Save back to file.
def extract_features(fname):

    # create an processed data folder
    if not os.path.exists('processed_data'):
        os.makedirs('processed_data')

    # get the data set from fname
    # has to use csv if the file is in csv format
    df = pd.read_csv(fname)

    # call the feature calculate functions in here
    df = moving_average(df,14)
    print(df.head)

    # save the modified excel file
    df.to_excel('processed_data/'+'sp500_processed.xlsx')



# TODO: construct EMA. (Gavin has some previous work on this.)
# def calc_SMA(df,win_size_list):

def moving_average(df,n):
    MA = pd.Series(ta.SMA(df['Close'],timeperiod=n), name='MA_' + str(n))
    df = df.join(MA)
    return df


def exponential_moving_average(df, n):
    """

    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    EMA = pd.Series(df['Close'].ewm(span=n, min_periods=n).mean(), name='EMA_' + str(n))
    df = df.join(EMA)
    return df


def momentum(df, n):
    """

    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    M = pd.Series(df['Close'].diff(n), name='MOM_' + str(n))
    df = df.join(M)
    return df


def rate_of_change(df, n):
    """

    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    # M = df['Close'].diff(n - 1)
    # N = df['Close'].shift(n - 1)
    # this function can also calculate n-day return

    ROC = pd.Series(df['Close'].pct_change(n)*100, name='ROC_' + str(n))
    df = df.join(ROC)
    return df


def average_true_range(df, n):
    """

    The average true range (ATR) is a technical analysis indicator that measures market volatility
    by decomposing the entire range of an asset price for that period.
    """
    # Result a little different from that of Yahoo Finance

    ATR = pd.Series(ta.ATR(df['High'], df['Low'], df['Close']), name='ATR_' + str(n))
    df = df.join(ATR)
    return df


def bollinger_bands(df, n):
    """

    A Bollinger Band® is a technical analysis tool defined by a set of lines plotted two standard deviations (positively and negatively)
     away from a simple moving average (SMA) of the security's pric
    """
    upperband, middleband, lowerband = ta.BBANDS(df['Close'], timeperiod=n, nbdevup=2, nbdevdn=2, matype=0)

    ub = pd.Series(upperband, name='ub_' + str(n))
    mb = pd.Series(middleband, name='mb_' + str(n))
    lb = pd.Series(lowerband, name='lb_' + str(n))

    df = df.join(ub)
    df = df.join(mb)
    df = df.join(lb)

    return df



def relative_strength_index(df, n):
    """Calculate Relative Strength Index(RSI) for given data.

    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    RSI = pd.Series(ta.RSI(df['Close'],n),name = 'RSI'+str(n))

    df = df.join(RSI)

    return df


def MACD(df):
    """
        The MACD is the difference between a 26-day and 12-day exponential moving average of closing prices.
        A 9-day EMA, called the "signal" line is plotted on top of the MACD to show buy/sell opportunities.
    """
    macd, macdsignal, macdhist = ta.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)

    Macd = pd.Series(macd, name='macd')
    Macdsignal = pd.Series(macdsignal, name='macdsignal')
    Macdhist = pd.Series(macdhist, name='macdhist')

    df = df.join(Macd).join(Macdsignal).join(Macdhist)
    return df


def william_r(df,n):
    """
    The Williams Percent Range, also called Williams %R, is a momentum indicator that shows you
    where the last closing price is relative to the highest and lowest prices of a given time period.
    """
    # real = WILLR(high, low, close, timeperiod=14)
    w_r = pd.Series(ta.WILLR(df['High'],df['Low'],df['Close'],timeperiod=n),name = "william_r_%" + str(n))
    df  = df.join(w_r)
    return df


def stocha_osc(df,n):
    """

    A stochastic oscillator is a momentum indicator comparing a particular closing price of a security
    to a range of its prices over a certain period of time.

    slowk, slowd = STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)

    %K is referred to sometimes as the slow stochastic indicator.
    The "fast" stochastic indicator is taken as %D = 3-period moving average of %K.

    """
    slowk,slowd = ta.STOCH(df['High'],df['Low'],df['Close'],fastk_period=n, slowk_period=3, slowk_matype=0, slowd_period=3,slowd_matype=0)

    k_line = pd.Series(slowk,name = "k%")
    d_line = pd.Series(slowd,name = "d%")

    return df.join(k_line).join(d_line)

def acc_dist(df):
    """
    Accumulation/distribution is a cumulative indicator that uses volume and price to assess whether a stock is being accumulated or distributed.
    The accumulation/distribution measure seeks to identify divergences between the stock price and volume flow
    This value is different from that of Yahoo Finance
    """


    CMFV = pd.Series(ta.AD(df['High'], df['Low'], df['Close'],df['Volume']), name="A/D - Current money flow volume")
    df = df.join(CMFV)
    return df


if __name__ == "__main__":
    fname = "original_data/" +"sp500.csv"
    extract_features(fname)