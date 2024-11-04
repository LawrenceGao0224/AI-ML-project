# filename: stock_price_data.py

import yfinance as yf

# Downloading historical data for Tesla and Meta
tesla_data = yf.download('TSLA', start='2022-01-01', end='2022-12-31')
meta_data = yf.download('META', start='2022-01-01', end='2022-12-31')

print(tesla_data.head())
print(meta_data.head())