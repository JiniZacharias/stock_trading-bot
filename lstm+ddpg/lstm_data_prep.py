import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Directory paths
models_dir = 'models'
data_dir = 'data'
output_dir = 'predictions'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# List of stock symbols (for demonstration, you can replace this with the actual list of S&P 500 symbols)
stock_symbols = ["AMZN", "AAPL","IBM", "NFLX", "NVDA", "ORCL"]

# Function to predict future prices for a given stock
def predict_future_prices(model, data, scaler, time_step, days=10):
    predictions = []
    last_data = data[-time_step:]
    for _ in range(days):
        last_data_scaled = scaler.transform(last_data)
        X_input = last_data_scaled.reshape(1, time_step, last_data_scaled.shape[1])
        pred_price = model.predict(X_input)
        predictions.append(pred_price[0][0])
        new_row = np.append(last_data[1:], [[pred_price[0][0], 0, 0, 0, 0]], axis=0)
        last_data = new_row
    placeholder = np.zeros((len(predictions), data.shape[1]))
    placeholder[:, 0] = predictions
    real_prices = scaler.inverse_transform(placeholder)[:, 0]
    real_prices = [float(price) if not np.isnan(price) else None for price in real_prices]
    return real_prices

# Function to categorize risk levels based on volatility
def categorize_risk_levels(data):
    data['Risk_Level'] = 'low'
    data.loc[data['Volatility'] > 0.02, 'Risk_Level'] = 'medium'
    data.loc[data['Volatility'] > 0.05, 'Risk_Level'] = 'high'
    return data

# Iterate over each stock symbol
for symbol in stock_symbols:
    try:
        # Load LSTM model and data for the stock
        model_path = os.path.join(models_dir, f'{symbol}_lstm_model.h5')
        data_path = os.path.join(data_dir, f'{symbol}.csv')
        
        lstm_model = load_model(model_path)
        data = pd.read_csv(data_path, parse_dates=['Date'], index_col='Date')
        data.fillna(method='ffill', inplace=True)
        data.fillna(method='bfill', inplace=True)
        data.dropna(inplace=True)

        # Prepare data for LSTM predictions
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data[['Close', 'SMA_10', 'SMA_50', 'RSI', 'Volume']])
        time_step = 100

        # Predict future prices for the next 10 days
        future_prices = predict_future_prices(lstm_model, data[['Close', 'SMA_10', 'SMA_50', 'RSI', 'Volume']].values, scaler, time_step, days=10)

        # Generate future dates
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=10, freq='B')  # 'B' frequency for business days

        # Create a DataFrame for future predictions
        future_data = pd.DataFrame({'Date': future_dates, 'LSTM_Prediction': future_prices})
        future_data.set_index('Date', inplace=True)

        # Calculate volatility and categorize risk levels
        future_data['Volatility'] = future_data['LSTM_Prediction'].pct_change().rolling(window=2).std()
        future_data = categorize_risk_levels(future_data)

        # Save the future predictions to a CSV file
        output_path = os.path.join(output_dir, f'{symbol}_future_predictions.csv')
        future_data.to_csv(output_path)

        print(f'Successfully saved predictions for {symbol} to {output_path}')
    
    except Exception as e:
        print(f'Error processing {symbol}: {e}')
