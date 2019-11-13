import quandl
quandl.ApiConfig.api_key = "BzmAGpzByrxtohyARK2B"

stock = input("Enter stock Symbol: ")

mydata = quandl.get("EIA/PET_RWTC_D")

data = quandl.get(f"EOD/{stock}")

print(data.head())
