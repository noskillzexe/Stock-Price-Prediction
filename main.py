import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
import threading

def get_stock_data(symbol, start_date, end_date):
    """
    Helper function to download stock data using yfinance library
    """
    return yf.download(symbol, start=start_date, end=end_date)

def download_data(symbols, start_date, end_date):
    """
    Function to download stock data for multiple symbols in parallel
    """
    data = {}
    with ThreadPoolExecutor() as executor:
        # Submit a thread for each symbol
        futures = {executor.submit(get_stock_data, symbol, start_date, end_date): symbol for symbol in symbols}
        # Collect the results as they become available
        for future in futures:
            symbol = futures[future]
            try:
                data[symbol] = future.result()
            except Exception as exc:
                print(f"Exception while downloading data for {symbol}: {exc}")
    return data

# Define the symbols to download data for
symbols = ['SPY']

# Define the start and end dates for the data
start_date = '2000-01-01'
end_date = '2026-01-01'

# Download the data for the symbols in parallel
data = download_data(symbols, start_date, end_date)

# Concatenate the data for all symbols into a single DataFrame
df = pd.concat([data[symbol] for symbol in symbols], axis=1)

# Filter data to only include closing prices
data = df.filter(['Close'])

# Convert data to numpy array
dataset = data.values

# Determine size of training set
training_data_len = int(np.ceil(len(dataset) * 0.8))

# Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

# Split data into training and testing sets
train_data = scaled_data[0:training_data_len, :]
test_data = scaled_data[training_data_len - 60: , :]

# Create training set
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

# Convert training set to numpy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)

# Reshape training set
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1, verbose=0)

# Create testing set
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

# Convert testing set to numpy array
x_test = np.array(x_test)

# Reshape testing set
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Get the model's predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
rmse = np.sqrt(np.mean(predictions - y_test)**2)
print(f"RMSE: {rmse}")

train = data[:training_data_len]
test = data[training_data_len:]
test.loc[:, 'Predictions'] = predictions

import matplotlib.pyplot as plt
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(test[['Close', 'Predictions']])
plt.legend(['Train', 'Test', 'Predictions'], loc='lower right')
plt.show()