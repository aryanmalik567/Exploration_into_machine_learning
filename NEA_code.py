import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import quandl

quandl.ApiConfig.api_key = "BzmAGpzByrxtohyARK2B"
stock = quandl.get("EOD/AAPL")

print(stock.head())
