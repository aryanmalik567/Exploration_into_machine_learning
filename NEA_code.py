import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import quandl

stockToRequest = input("Enter stock symbol of stock you would like to fetch: ")

quandl.ApiConfig.api_key = "BzmAGpzByrxtohyARK2B"
stock = quandl.get(f"EOD/{stockToRequest}")
stock = pd.DataFrame(stock)

neededColumns = ['Date', 'Adj_Close', 'Adj_Volume']
stock1 = stock[neededColumns]

print(stock1.head())
