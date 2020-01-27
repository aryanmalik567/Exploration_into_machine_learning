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

neededColumns = ['Adj_Close']  # Add in volume later
stock = stock[neededColumns]

'''
# Visualizing our data to begin with
plt.plot(stock)
plt.xlabel('Date')
plt.ylabel('AdjClose')
plt.show()
'''

# Creating 80% train split
numRows = stock.shape[0]
first80pct = round(0.8 * numRows)

stockTrain = stock[:first80pct]
stockTrainMean = stockTrain.mean()
stockTrainStd = stockTrain.std()

stockTrainNormalized = (stockTrain - stockTrainMean) / stockTrainStd

# print(stockTrainNormalized.shape) gives (872, 1)

# Creating 20% testing split
last20pct = round(0.2 * numRows)

stockTest = stock[first80pct:last20pct]
stockTestMean = stockTest.mean()
stockTestStd = stockTest.std()

stockValNormalized = (stockTest - stockTestMean) / stockTestStd

dataPoints = 300  # 300 time steps worth of training data for each sequence
futurePeriodPrediction = 30  # Predicting 30 time steps into the future in each sequence

x_train = []  # List containing several sequences each of time step length 300
y_train = []  # List containing several sequences each of time step length 30

for i in range(dataPoints, len(stockTrainNormalized)):
    x_train.append(stockTrainNormalized[i - dataPoints:i])
    y_train.append(stockTrainNormalized[i, 0])



