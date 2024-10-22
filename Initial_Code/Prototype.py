import pandas as pd
from collections import deque
import random
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
import time
from sklearn import preprocessing

dataPoints = 100  # 100 days of data
futurePeriodPrediction = 14  # Predict 2 weeks into the future
stockToPredict = "SPY"
epochs = 10
batchSize = 64
name = f"{dataPoints}-Sequence-{futurePeriodPrediction}-Prediction-{int(time.time())}"


def classify(current, future):
    if float(future) > float(current):  # If stock price has increased, return positive indicator (1)
        return 1
    else:                               # If stock price has decreased, return positive indicator (0)
        return 0


def preprocessor(df):
    df = df.drop("future", 1)  # Remove actual data so ML model doesn't use this

    for col in df.columns:  # Iterate through each column
        if col != "target":  # Provided the column is not the target column
            df[col] = df[col].pct_change()  # Replace each column with data that represents the percentage change in
            # price from the previous data point
            df.dropna(inplace=True)  # Remove any NaN values
            df[col] = preprocessing.scale(df[col].values)  # Scale each value to between 0 and 1

    df.dropna(inplace=True)  # Remove any additional NaN's

    #print(df)

    sequentialData = []  # Contains all sequences
    prevDays = deque(maxlen=dataPoints)  # Shortens whole sequence to only 5 years

    for i in df.values:
        prevDays.append([n for n in i[:-1]])  # Iterate through all columns except target
        if len(prevDays) == dataPoints:
            sequentialData.append([np.array(prevDays), i[-1]])

    random.shuffle(sequentialData)

    buys = []
    sells = []

    for seq, target in sequentialData:
        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])

    random.shuffle(buys)
    random.shuffle(sells)

    lower = min(len(buys), len(sells))

    buys = buys[:lower]
    sells = sells[:lower]  # Therefore number of buys and sells are now equal

    sequentialData = buys + sells
    random.shuffle(sequentialData)  # Very important shuffle otherwise data will be a series of buys followed by sells

    X = []
    y = []

    for seq, target in sequentialData:
        X.append(seq)
        y.append(target)

    X = np.array(X)

    return X, y


main_df = pd.DataFrame()

stocks = ["SPY", "GOOG", "XOM", "GLD"]

for stock in stocks:
    dataSet = f'Data/{stock}.csv'

    df = pd.read_csv(dataSet)
    df.rename(columns={"Adj Close": f"{stock}_close", "Volume": f"{stock}_volume"}, inplace=True)

    df.set_index("Date", inplace=True)
    df = df[[f'{stock}_close', f'{stock}_volume']]

    if len(main_df) == 0:
        main_df = df
    else:
        main_df = main_df.join(df)


main_df.fillna(method="ffill", inplace=True)  # if there are gaps in data, use previously known values
main_df.dropna(inplace=True)

main_df['future'] = main_df[f'{stockToPredict}_close'].shift(-futurePeriodPrediction)
main_df['target'] = list(map(classify, main_df[f"{stockToPredict}_close"], main_df["future"]))

main_df.dropna(inplace=True)

# print(main_df[[f"{stockToPredict}_close", "future", "target"]].head(20))

dates = main_df.index.values  # Isolating the dates column
last10pct = sorted(main_df.index.values)[-int(0.1*len(dates))]  # Find the date that is 10 percent of the way through
# the entire set of dates

actualData = main_df[(main_df.index >= last10pct)]  # Isolate the last 5% data from rest of data frame
main_df = main_df[main_df.index < last10pct]  # Adjust main data frame to remove data for these dates

# print(actualData)

train_x, train_y = preprocessor(main_df)
actual_x, actual_y = preprocessor(actualData)

print(f"train data: {len(train_x)} validation: {len(actual_x)}")
print(f"Don't buys: {train_y.count(0)}, buys: {train_y.count(1)}")
print(f"VALIDATION Don't buys: {actual_y.count(0)}, buys: {actual_y.count(1)}")

model = Sequential()
model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(2, activation="softmax"))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=opt,
              metrics=['accuracy'])

tensorboard = TensorBoard(log_dir=f'logs/{name}')

filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"  # unique file name that will include the epoch and the
# validation acc for that epoch
checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1,
                                                      save_best_only=True, mode='max'))  # saves only the best ones

history = model.fit(
    train_x, train_y,
    batch_size=128,
    epochs=epochs,
    validation_data=(actual_x, actual_y),
    callbacks=[tensorboard, checkpoint],
)

model.add(keras.layers.Flatten())

score = model.evaluate(actual_x, actual_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save("models/{}".format(name))
