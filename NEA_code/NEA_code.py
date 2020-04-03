import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import quandl
from pandas.plotting import register_matplotlib_converters

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
register_matplotlib_converters()
quandl.ApiConfig.api_key = "BzmAGpzByrxtohyARK2B"


modelTrial = 1  # First model


def const_selector(model_number):
    epochs_list = [10, 20, 30, 40]
    epochSteps_list = [40, 30, 20, 10]

    return epochs_list[model_number], epochSteps_list[model_number]


class Constants:
    def __init__(self, past_period, future_period, batch_size, buffer_size, epochs, epoch_steps, val_steps):
        self.past_period = past_period
        self.future_period = future_period
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.epochs = epochs
        self.epoch_steps = epoch_steps
        self.val_steps = val_steps


consts = Constants(300, 30, 300, 10000, const_selector()[0], const_selector()[1], 10)


def quandl_request():
    # stockToRequest = input("Enter stock symbol of stock you would like to fetch: ")
    # stock = quandl.get(f"EOD/{stockToRequest}")
    stock = quandl.get(f"EOD/MSFT")
    stock = pd.DataFrame(stock)
    needed_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    stock = stock[needed_columns]

    return stock


def data_split(stock):
    # Creating training split
    numRows = stock.shape[0]
    trainSplit = numRows - consts.future_period - consts.past_period

    stockTrain = stock[:trainSplit]
    stockTrainMean = stockTrain.mean()
    stockTrainStd = stockTrain.std()

    stockTrainStandardized = (stockTrain - stockTrainMean) / stockTrainStd
    stockTrainStandardized = np.array(stockTrainStandardized)

    # Creating testing split
    stockTest = stock[trainSplit:numRows]
    stockTestMean = stockTest.mean()
    stockTestStd = stockTest.std()

    stockTestStandardized = (stockTest - stockTestMean) / stockTestStd
    stockTestStandardized = np.array(stockTestStandardized)

    return stockTrainStandardized, stockTestStandardized


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


def x_y(stock_train, stock_test):
    x_train, y_train = sequencer(stock_train, consts.past_period, consts.future_period)

    np.swapaxes(x_train, 0, 2)

    x_test = stock_test[:consts.past_period]  # First 300 time steps of test sequence

    stock_test = stock_test[:, 4]
    y_test = stock_test[consts.past_period:(consts.past_period + consts.future_period)]

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    x_test = np.reshape(x_test, (1, 300, 5))
    y_test = np.reshape(y_test, (1, 30))

    return x_train, y_train, x_test, y_test


def plot_output(history, actual_future, prediction):
    plt.figure(figsize=(12, 6))
    history_range = list(range(-len(history[:, 4]), 0))
    future_range = list(range(0, len(actual_future)))

    plt.plot(history_range, np.array(history[:, 4]), label='History')
    plt.plot(future_range, np.array(actual_future), color='cyan', label='Actual Future')
    plt.plot(future_range, np.array(prediction), color='red', label='Prediction')

    plt.legend(loc='upper left')
    plt.show()


def train_model(x_train, y_train, x_test, y_test):
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = train_data.cache().batch(consts.batch_size).repeat()

    test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_data = test_data.batch(consts.batch_size).repeat()

    stockModel = tf.keras.models.Sequential()
    stockModel.add(tf.keras.layers.LSTM(32, return_sequences=True, input_shape=x_train.shape[-2:]))
    stockModel.add(tf.keras.layers.LSTM(16, activation='relu'))
    stockModel.add(tf.keras.layers.Dense(30))

    # stockModel.compile(optimizer='adam', loss='mean_squared_error')
    stockModel.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')
    stockModelHistory = stockModel.fit(train_data, epochs=consts.epochs, steps_per_epoch=consts.epoch_steps, validation_data=test_data, validation_steps=consts.val_steps)

    # plot_output(x_test, y_test, stockModel.predict(x_test))

    for x, y in test_data.take(1):
        plot_output(x[0], y[0], stockModel.predict(x)[0])


def main():
    stock = quandl_request()
    train_test = data_split(stock)
    xy_arrays = x_y(train_test[0], train_test[1])
    train_model(xy_arrays[0], xy_arrays[1], xy_arrays[2], xy_arrays[3])


main()



