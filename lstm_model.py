'''import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Ensure 'models' folder exists to save the trained models
os.makedirs("models", exist_ok=True)

# Prepare the data for LSTM
def create_dataset(data, time_step=1, prediction_days=10):
    X, Y = [], []
    for i in range(len(data) - time_step - prediction_days):
        a = data[i:(i + time_step), :]
        X.append(a)
        Y.append(data[i + time_step:i + time_step + prediction_days, 0])
    return np.array(X), np.array(Y)

# Get the list of stock symbols from the data folder
data_folder = "data"
#symbols = [file.split(".")[0] for file in os.listdir(data_folder) if file.endswith(".csv")]
symbols=["CSCO"]

time_step = 100
prediction_days = 10

for symbol in symbols:
    # Load and preprocess the data
    file_path = os.path.join(data_folder, f"{symbol}.csv")
    data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

    # Display the first few rows of the raw data
    print(f"Raw data for {symbol}:")
    print(data.head())

    # Check for NaN values in the raw data
    if np.any(np.isnan(data)):
        print(f"NaN values found in raw data for {symbol}")
        data.fillna(method='ffill', inplace=True)  # Forward fill NaN values
        data.fillna(method='bfill', inplace=True)  # Backward fill NaN values

    # Drop any remaining NaN values
    data.dropna(inplace=True)

    # Normalize the data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[['Close', 'SMA_10', 'SMA_50', 'RSI', 'Volume']])

    # Display the first few rows of the scaled data
    print(f"Scaled data for {symbol}:")
    print(pd.DataFrame(data_scaled, columns=['Close', 'SMA_10', 'SMA_50', 'RSI', 'Volume']).head())

    # Check for NaN values in the scaled data
    if np.any(np.isnan(data_scaled)):
        print(f"NaN values found in scaled data for {symbol}")

    X, Y = create_dataset(data_scaled, time_step, prediction_days)

    # Display the first few rows of the dataset
    print(f"Dataset for {symbol}:")
    print(f"X: {X[:5]}")
    print(f"Y: {Y[:5]}")

    # Check for NaN values in the dataset
    if np.any(np.isnan(X)) or np.any(np.isnan(Y)):
        print(f"NaN values found in dataset for {symbol}")

    # Reshape input to be [samples, time steps, features]
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2])

    # Create the LSTM model
    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=(time_step, X.shape[2])))
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(50))
    model.add(Dense(prediction_days))  # Output layer with neurons for 10 days prediction

    # Compile the model with gradient clipping
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipvalue=1.0)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Train the model with KeyboardInterrupt handling
    try:
        model.fit(X, Y, epochs=100, batch_size=32)
    except KeyboardInterrupt:
        print("Training interrupted by user.")

    # Save the model
    model.save(os.path.join("models", f"{symbol}_lstm_model.h5"))
    print(f"✅ LSTM model for {symbol} trained and saved.")

    # Evaluate the model
    predictions = model.predict(X)
    mae = mean_absolute_error(Y, predictions)
    rmse = np.sqrt(mean_squared_error(Y, predictions))
    print(f"Model performance for {symbol}: MAE = {mae}, RMSE = {rmse}") '
    '''

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Ensure 'models' folder exists to save the trained models
os.makedirs("models", exist_ok=True)

# Prepare the data for LSTM
def create_dataset(data, time_step=1, prediction_days=10):
    X, Y = [], []
    for i in range(len(data) - time_step - prediction_days):
        a = data[i:(i + time_step), :]
        X.append(a)
        Y.append(data[i + time_step:i + time_step + prediction_days, 0])
    return np.array(X), np.array(Y)

# Specify the stock symbol and file path
#["AMZN", "AAPL","IBM", "NFLX", "NVDA", "ORCL"]

symbol = "ORCL"
file_path = os.path.join("data", f"{symbol}.csv")

time_step = 100
prediction_days = 10

# Load and preprocess the data
data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

# Display the first few rows of the raw data
print(f"Raw data for {symbol}:")
print(data.head())

# Check for NaN values in the raw data
if np.any(np.isnan(data)):
    print(f"NaN values found in raw data for {symbol}")
    data.fillna(method='ffill', inplace=True)  # Forward fill NaN values
    data.fillna(method='bfill', inplace=True)  # Backward fill NaN values

# Drop any remaining NaN values
data.dropna(inplace=True)

# Normalize the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[['Close', 'SMA_10', 'SMA_50', 'RSI', 'Volume']])

# Display the first few rows of the scaled data
print(f"Scaled data for {symbol}:")
print(pd.DataFrame(data_scaled, columns=['Close', 'SMA_10', 'SMA_50', 'RSI', 'Volume']).head())

# Check for NaN values in the scaled data
if np.any(np.isnan(data_scaled)):
    print(f"NaN values found in scaled data for {symbol}")

X, Y = create_dataset(data_scaled, time_step, prediction_days)

# Display the first few rows of the dataset
print(f"Dataset for {symbol}:")
print(f"X: {X[:5]}")
print(f"Y: {Y[:5]}")

# Check for NaN values in the dataset
if np.any(np.isnan(X)) or np.any(np.isnan(Y)):
    print(f"NaN values found in dataset for {symbol}")

# Reshape input to be [samples, time steps, features]
X = X.reshape(X.shape[0], X.shape[1], X.shape[2])

# Create the LSTM model
model = Sequential()
model.add(Masking(mask_value=0.0, input_shape=(time_step, X.shape[2])))
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dense(prediction_days))  # Output layer with neurons for 10 days prediction

# Compile the model with gradient clipping
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipvalue=1.0)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Train the model with KeyboardInterrupt handling
try:
    model.fit(X, Y, epochs=100, batch_size=32)
except KeyboardInterrupt:
    print("Training interrupted by user.")

# Save the model
model.save(os.path.join("models", f"{symbol}_lstm_model.h5"))
print(f"✅ LSTM model for {symbol} trained and saved.")

# Evaluate the model
predictions = model.predict(X)
mae = mean_absolute_error(Y, predictions)
rmse = np.sqrt(mean_squared_error(Y, predictions))
print(f"Model performance for {symbol}: MAE = {mae}, RMSE = {rmse}")
