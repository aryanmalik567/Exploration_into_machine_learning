import pandas as pd

main_df = pd.DataFrame()

stocks = ["SPY", "GOOG", "XOM", "GLD"]

for stock in stocks:
    dataset = f'Data/{stock}.csv'

    df = pd.read_csv(dataset)
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

print(main_df[[f"{stockToPredict}_close", "future", "target"]].head(10))
