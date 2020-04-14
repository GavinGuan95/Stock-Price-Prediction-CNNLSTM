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
import glob
# import talib as ta

# import still works even though the error message persists
import technical_indicators as ti

# TODO: derive adjusted Open, High, Low from the downloaded data. Save back to file.
def extract_features(fname, start=None, end=None, index="Date", economic=True):

    # create an processed data folder
    if not os.path.exists('processed_data'):
        os.makedirs('processed_data')

    # get the data set from fname
    # has to use csv if the file is in csv format
    df = pd.read_csv(fname)
    df = df.set_index(index)

    # select a time frame
    if start is None and end is None:
        df = df
    elif start is None and end is not None:
        df = df.loc[:end]
    elif start is not None and end is None:
        df = df.loc[start:]
    else:
        df = df[start:end]

    # call the technical functions in here
    df = ti.moving_average(df, 2, 3)
    df = ti.exponential_moving_average(df, 2, 3)
    df = ti.moving_average(df, 2, 5)
    df = ti.exponential_moving_average(df, 2, 5)
    df = ti.moving_average(df, 2, 10)
    df = ti.exponential_moving_average(df, 2, 10)
    df = ti.moving_average(df, 2, 20)
    df = ti.exponential_moving_average(df, 2, 20)
    df = ti.rate_of_change(df, 1)

    # call the economic functions in here
    if economic:
        df = append_indices(df)

    print(df.head)
    # save the modified excel file
    df.to_csv("/home/guanyush/Pictures/CSC2516/CNNLSTM/data_loader/original_data/SPY2.csv")


def append_indices(df):
    # this function will append all the indices from the indices folder
    for fname in glob.glob('original_data/indices/*.csv'):
        print(fname)
        df_i = pd.read_csv(fname)
        df = match(df, df_i, fname.split("\\")[1].split(".")[0])
        print(df_i.shape,df.shape)
    return df


# helper function for aligning feature data against the data of interest in this case SPY
def match(sp_df, df_i, nam):
    # fill missing data by extending previous day's day
    df_i = df_i.fillna(method='pad')

    # select only the closing price for further processing
    df_i = df_i.loc[:, ["Date", "Close"]]
    df_i = df_i.rename(columns={"Close": nam})

    # merge the usd index into s&p500 based on the 'Date' column
    # The method will remove the entries in usd index that did not appear in s&p500

    df_merged = pd.merge(sp_df, df_i, how="inner", on=["Date"])

    if df_merged.shape[0]<sp_df.shape[0]:
        df_merged = pd.merge(sp_df, df_merged.loc[:, ["Date", nam]], how="outer", on=["Date"])
        df_merged = df_merged.fillna(method ="pad")

    return df_merged


if __name__ == "__main__":
    # fname = "/home/guanyush/Pictures/CSC2516/CNNLSTM/data_loader/processed_data/kibot.csv"
    # extract_features(fname, index="Unnamed: 0", economic=False)

    fname = os.path.join("/home/guanyush/Pictures/CSC2516/CNNLSTM/data_loader/original_data/SPY.csv")
    extract_features(fname, economic=False)