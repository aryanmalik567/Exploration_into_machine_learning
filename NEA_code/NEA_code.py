import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
'''
plt.plot(stock['Close'])
plt.xlabel('Date')
plt.ylabel('Close')
plt.show()
'''

# Creating train split
numRows = stock.shape[0]
trainSplit = numRows - futurePeriodPrediction - dataPoints

stockTrain = stock[:trainSplit]
stockTrainMean = stockTrain.mean()
stockTrainStd = stockTrain.std()

stockTrainStandardized = (stockTrain - stockTrainMean) / stockTrainStd

stockTrainStandardized = np.array(stockTrainStandardized)

# print(stockTrainStandardized.shape) gives (760, 5)

# Creating testing split
stockTest = stock[trainSplit:numRows]
stockTestMean = stockTest.mean()
stockTestStd = stockTest.std()

stockTestStandardized = (stockTest - stockTestMean) / stockTestStd

stockTestStandardized = np.array(stockTestStandardized)

# print(stockTestNormalized.shape) gives (330, 5)
'''
x_train = []  # List containing several sequences each of time step length 300, training data
y_train = []  # List containing several sequences each of time step length 30, validation data
'''

'''
for x in range(dataPoints, (len(stockTrainStandardized) - futurePeriodPrediction)):  # From 300 to 730
    x_train.append(stockTrainStandardized[x - dataPoints:x])  # Append 300 time steps from range 0 to 460

for y in range(dataPoints, (len(stockTrainStandardized) - futurePeriodPrediction)):  # From 300 to 730
    y_train.append(stockTrainStandardized[y:y + futurePeriodPrediction])  # Append 30 time steps from range 300 to 730
'''


def sequencer(dataset, history_size, target_size):
    training_data = []
    target_data = []

    target_dataset = dataset[:, 4]

    for i in range(history_size, (len(dataset) - target_size)):
        indices_x = range(i-history_size, i)
        training_data.append(dataset[indices_x])

        indices_y = range(i, i + target_size)
        target_data.append(target_dataset[indices_y])

    return np.array(training_data), np.array(target_data)


x_train, y_train = sequencer(stockTrainStandardized, dataPoints, futurePeriodPrediction)

np.swapaxes(x_train, 0, 2)
# np.swapaxes(y_train, 0, 2)

x_test = stockTestStandardized[:dataPoints]  # First 300 time steps of test sequence

stockTestStandardized = stockTestStandardized[:, 4]
y_test = stockTestStandardized[dataPoints:(dataPoints + futurePeriodPrediction)]

x_test = np.array(x_test)
y_test = np.array(y_test)

x_test = np.reshape(x_test, (1, 300, 5))
y_test = np.reshape(y_test, (1, 30))

BATCH_SIZE = 300
BUFFER_SIZE = 10000

EpochSteps = 50
EPOCHS = 20


def plot_output(history, actual_future, prediction):
    plt.figure(figsize=(12, 6))
    history_range = list(range(-len(history[:, 4]), 0))
    future_range = list(range(0, len(actual_future)))

    plt.plot(history_range, np.array(history[:, 4]), label='History')
    plt.plot(future_range, np.array(actual_future), color='cyan', label='Actual Future')
    plt.plot(future_range, np.array(prediction), color='red', label='Prediction')

    plt.legend(loc='upper left')
    plt.show()


train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.cache().batch(BATCH_SIZE).repeat()

test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_data = test_data.batch(BATCH_SIZE).repeat()

stockModel = tf.keras.models.Sequential()
stockModel.add(tf.keras.layers.LSTM(32, return_sequences=True, input_shape=x_train.shape[-2:]))
stockModel.add(tf.keras.layers.LSTM(16, activation='relu'))
stockModel.add(tf.keras.layers.Dense(30))

# stockModel.compile(optimizer='adam', loss='mean_squared_error')
stockModel.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')
stockModelHistory = stockModel.fit(train_data, epochs=EPOCHS, steps_per_epoch=EpochSteps, validation_data=test_data, validation_steps=10)

# plot_output(x_test, y_test, stockModel.predict(x_test))

for x, y in test_data.take(1):
    plot_output(x[0], y[0], stockModel.predict(x)[0])



