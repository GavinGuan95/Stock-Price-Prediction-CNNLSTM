"""
Download stock data only.
"""
import os
import pandas as pd
import yfinance as yf

csv_root = "../data/STOCK"
os.mkdir(csv_root)

ticker_list = ["SPY"]
start_date = '2019-01-01'
end_date = '2019-12-31'

for ticker in ticker_list:
    df = yf.download(ticker,
                     start=start_date,
                     end=end_date,
                     progress=False)
    csv_file_path = os.path.join(csv_root, ticker + ".csv")
    df.to_csv(csv_file_path)
