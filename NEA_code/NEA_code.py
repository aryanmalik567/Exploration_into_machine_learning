import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import quandl

stockToRequest = input("Enter stock symbol of stock you would like to fetch: ")

stock = quandl.get(f"EOD/{stockToRequest}", api_key="BzmAGpzByrxtohyARK2B")
stock = stock[["Close"]]

print(stock.shape)
