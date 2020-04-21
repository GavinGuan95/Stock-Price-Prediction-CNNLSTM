"""
Download stock data only.
"""
import os
import pandas as pd
import yfinance as yf

csv_root = "../data_loader/original_data/indices"
if not os.path.exists(csv_root):
    os.mkdir(csv_root)

# index_list = ["SPY","DJI","^IXIC","^GSPC","^NYA",
#                "^RUT","^HSI","000001.SS","^FCHI","^FTSE",
#                "^GDAXI"]

index_list = []

# company_list = ["AAPL","AMZN","GE","JNJ","JPM","MSFT","WFC","XOM"]
company_list = ["TSLA"]


# forex_list = ["JPY=X","CNY=X","AUD=X","CAD=X","CHF=X","EUR=X","GBP=X","NZD=X"]
forex_list = []
ticker_list = index_list + forex_list + company_list

start_date = '2000-01-01'
end_date = '2020-04-01'

for ticker in ticker_list:
    df = yf.download(ticker,
                     start=start_date,
                     end=end_date,
                     progress=False)
    csv_file_path = os.path.join(csv_root, ticker + ".csv")
    df.to_csv(csv_file_path)
