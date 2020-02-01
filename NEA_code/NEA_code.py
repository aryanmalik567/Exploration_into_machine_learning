import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import quandl
import tensorflow as tf

# stockToRequest = input("Enter stock symbol of stock you would like to fetch: ")

quandl.ApiConfig.api_key = "BzmAGpzByrxtohyARK2B"
# stock = quandl.get(f"EOD/{stockToRequest}")
stock = quandl.get(f"EOD/AAPL")
stock = pd.DataFrame(stock)

neededColumns = ['Adj_Close']  # Add in volume later
stock = stock[neededColumns]

dataPoints = 300  # 300 time steps worth of training data for each sequence
futurePeriodPrediction = 30  # Predicting 30 time steps into the future in each sequence

'''
# Visualizing our data to begin with
plt.plot(stock)
plt.xlabel('Date')
plt.ylabel('AdjClose')
plt.show()
'''

# Creating train split
numRows = stock.shape[0]
trainSplit = numRows - futurePeriodPrediction - dataPoints

stockTrain = stock[:trainSplit]
stockTrainMean = stockTrain.mean()
stockTrainStd = stockTrain.std()

stockTrainStandardized = (stockTrain - stockTrainMean) / stockTrainStd

# print(stockTrainStandardized.shape) gives (760, 1)

# Creating testing split
stockTest = stock[trainSplit:numRows]
stockTestMean = stockTest.mean()
stockTestStd = stockTest.std()

stockTestNormalized = (stockTest - stockTestMean) / stockTestStd

# print(stockTestNormalized.shape) gives (330, 1)

x_train = []  # List containing several sequences each of time step length 300, training data
y_train = []  # List containing several sequences each of time step length 30, validation data

for x in range(dataPoints, (len(stockTrainStandardized) - futurePeriodPrediction)):  # From 300 to 730
    x_train.append(stockTrainStandardized[x - dataPoints:x])  # Append 300 time steps from range 0 to 460

for y in range(dataPoints, (len(stockTrainStandardized) - futurePeriodPrediction)):  # From 30 to 760
    y_train.append(stockTrainStandardized[y:y + futurePeriodPrediction])  # Append 30 time steps from range 30 to 490

BATCH_SIZE = 256
BUFFER_SIZE = 10000


