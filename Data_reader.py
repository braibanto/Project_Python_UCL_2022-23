import yfinance as yf
import datetime as dt
import pandas_datareader as pdr

tickers = ['AAPL', 'MSFT', 'AMD', 'IBM']
start = dt.datetime(2020, 1, 1)
stop = dt.datetime(2021, 1, 1)
#data = yf.download(tickers, start=start, end=stop, progress=True)

data = pdr.DataReader(tickers , data_source='stooq', start=start, end=stop)
data = data[::-1]
data_c = data['Close']

print(data_c.index)


