# filename: stock_price_chart.py
import matplotlib.pyplot as plt

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