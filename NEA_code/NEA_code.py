import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import quandl
import tensorflow as tf
import random
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# stockToRequest = input("Enter stock symbol of stock you would like to fetch: ")

quandl.ApiConfig.api_key = "BzmAGpzByrxtohyARK2B"
# stock = quandl.get(f"EOD/{stockToRequest}")
stock = quandl.get(f"EOD/MSFT")
stock = pd.DataFrame(stock)

neededColumns = ['Open', 'High', 'Low', 'Close', 'Volume']
stock = stock[neededColumns]

dataPoints = 300  # 300 time steps worth of training data for each sequence
futurePeriodPrediction = 30  # Predicting 30 time steps into the future in each sequence

# Visualizing our data to begin with
plt.plot(stock['Close'])
plt.xlabel('Date')
plt.ylabel('Close')
# plt.show()

# Creating train split
numRows = stock.shape[0]
trainSplit = numRows - futurePeriodPrediction - dataPoints

stockTrain = stock[:trainSplit]
stockTrainMean = stockTrain.mean()
stockTrainStd = stockTrain.std()

stockTrainStandardized = (stockTrain - stockTrainMean) / stockTrainStd

# print(stockTrainStandardized.shape) gives (760, 5)

# Creating testing split
stockTest = stock[trainSplit:numRows]
stockTestMean = stockTest.mean()
stockTestStd = stockTest.std()

stockTestNormalized = (stockTest - stockTestMean) / stockTestStd

# print(stockTestNormalized.shape) gives (330, 5)

x_train = []  # List containing several sequences each of time step length 300, training data
y_train = []  # List containing several sequences each of time step length 30, validation data

for x in range(dataPoints, (len(stockTrainStandardized) - futurePeriodPrediction)):  # From 300 to 730
    x_train.append(stockTrainStandardized[x - dataPoints:x])  # Append 300 time steps from range 0 to 460

# x_train = np.array(x_train)

for y in range(dataPoints, (len(stockTrainStandardized) - futurePeriodPrediction)):  # From 30 to 760
    y_train.append(stockTrainStandardized[y:y + futurePeriodPrediction])  # Append 30 time steps from range 30 to 490

x_test = stockTestNormalized[:dataPoints]  # First 300 time steps of test sequence
y_test = stockTestNormalized[dataPoints:(dataPoints + futurePeriodPrediction)]

BATCH_SIZE = 300
BUFFER_SIZE = 10000

'''
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_data = test_data.batch(BATCH_SIZE).repeat()

stockModel = tf.keras.models.Sequential()
stockModel.add(tf.keras.layers.LSTM(32, return_sequences=True, input_shape=x_train.shape[-2:]))
stockModel.add(tf.keras.layers.LSTM(16, activation='relu'))
stockModel.add(tf.keras.layers.Dense(30))

stockModel.compile(optimizer='adam', loss='mean_squared_error')
stockModel.fit(x_train, y_train, epochs=100, batch_size=BATCH_SIZE)
'''

print(len(x_train))
print(len(x_train[1][1]))

