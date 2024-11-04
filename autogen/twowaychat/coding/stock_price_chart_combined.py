# filename: stock_price_chart_combined.py
import yfinance as yf
import matplotlib.pyplot as plt

# Downloading historical data for Tesla and Meta
tesla_data = yf.download('TSLA', start='2022-01-01', end='2022-12-31')
meta_data = yf.download('META', start='2022-01-01', end='2022-12-31')

# Plotting the adjusted closing prices for Tesla and Meta
plt.figure(figsize=(14, 7))
plt.plot(tesla_data['Adj Close'], label='Tesla', color='blue')
plt.plot(meta_data['Adj Close'], label='Meta', color='orange')

plt.title('Tesla vs Meta Stock Price Change (2022)')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.show()