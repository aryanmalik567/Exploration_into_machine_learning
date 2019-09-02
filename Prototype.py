import pandas as pd
from sklearn import preprocessing
from collections import deque
import numpy as np
import random

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


dataPoints = (365-104) * 5  # Number of days in a year without weekends x 5
futurePeriodPrediction = 30  # Predict a month into the future
stockToPredict = "SPY"


def classify(current, future):
    if float(future) > float(current):  # If stock price has increased, return positive indicator (1)
        return 1
    else:                               # If stock price has decreased, return positive indicator (0)
        return 0


main_df['future'] = main_df[f'{stockToPredict}_close'].shift(-futurePeriodPrediction)

main_df['target'] = list(map(classify, main_df[f"{stockToPredict}_close"], main_df["future"]))

# print(main_df[[f"{stockToPredict}_close", "future", "target"]].head(20))

dates = main_df.index.values  # Isolating the dates column
last5pct = dates[-int(0.05*len(dates))]  # Find the date that is 5 percent of the way through the entire set of dates

actualData = main_df[(main_df.index >= last5pct)]  # Isolate the last 5% data from rest of data frame
main_df = main_df[main_df.index < last5pct]  # Adjust main data frame to remove data for these dates


def preprocessor(df):
    df = df.drop("future", 1)  # Remove actual data so ML model doesn't use this

    for col in df.columns:  # Iterate through each column
        if col != "target":  # Provided the column is not the target column
            df[col] = df[col].pct_change()  # Replace each column with data that represents the percentage change in
            # price from the previous data point
            df.dropna(inplace=True)  # Remove any NaN values
            df[col] = preprocessing.scale(df[col].values)  # Scale each value to between 0 and 1

    df.dropna(inplace=True)  # Remove any additional NaN's

    sequentialData = []
    prevDays = deque(maxlen=dataPoints)  # Creates a que with max len of 5 years worth of data

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
    Y = []

    for seq, target in sequentialData:
        X.append(seq)
        Y.append(target)


preprocessor(main_df)
#train
