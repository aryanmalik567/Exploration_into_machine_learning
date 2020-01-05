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

neededColumns = ['Adj_Close']  # Just uni-variate to begin with
stock = stock[neededColumns]

print(stock.head())
