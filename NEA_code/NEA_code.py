import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import quandl

# stockToRequest = input("Enter stock symbol of stock you would like to fetch: ")

quandl.ApiConfig.api_key = "BzmAGpzByrxtohyARK2B"
# stock = quandl.get(f"EOD/{stockToRequest}")
stock = quandl.get(f"EOD/AAPL")
stock = pd.DataFrame(stock)

neededColumns = ['Adj_Close', 'Adj_Volume']
stock = stock[neededColumns]

# Creating 60% train split
numRows = stock.shape[0]
first60pct = round(0.6 * numRows)

stockTrainStd = stock[:first60pct].mean()
stockTrainMean = stock[:first60pct].std()

stockNormalized = (stock - stockTrainMean) / stockTrainStd






