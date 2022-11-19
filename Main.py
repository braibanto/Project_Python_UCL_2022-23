import pandas_datareader as pdr
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

start = dt.datetime(2020, 1, 1)
stop = dt.datetime(2021, 1, 1)
data_df = pdr.get_data_yahoo("MSFT", start, stop)
arr = data_df.to_numpy()

print(data_df.head())
print("Moyenne Low: ", data_df["Low"].mean(), "  ",
      "Moyenne High: ", data_df["High"].mean()
      )
print(data_df[["High", "Low"]].describe())
