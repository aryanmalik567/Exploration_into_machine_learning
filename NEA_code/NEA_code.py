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

stockTrainStandardized = (stockTrain - stockTrainMean) / stockTrainStd

# print(stockTrainStandardized.shape) gives (872, 1)

# Creating 20% testing split
last20pct = round(0.2 * numRows)

stockVal = stock[first80pct:last20pct]
stockValMean = stockVal.mean()
stockValStd = stockVal.std()

stockValNormalized = (stockVal - stockValMean) / stockValStd

dataPoints = 300  # 300 time steps worth of training data for each sequence
futurePeriodPrediction = 30  # Predicting 30 time steps into the future in each sequence

x_train = []  # List containing several sequences each of time step length 300, training data
y_train = []  # List containing several sequences each of time step length 30, validation data

for x in range(dataPoints, (len(stockTrainStandardized) - futurePeriodPrediction)):  # From 300 to 842
    x_train.append(stockTrainStandardized[x - dataPoints:x])  # Append 300 time steps from range 0 to 542

for y in range(dataPoints, (len(stockTrainStandardized) - futurePeriodPrediction)):  # From 30 to 842
    y_train.append(stockTrainStandardized[y:y + futurePeriodPrediction])  # Append 30 time steps from range 30 to 572

print(len(x_train))
print(len(y_train))

print(len(x_train[0]))
print(len(y_train[0]))

print(x_train[541])
print(y_train[541])



