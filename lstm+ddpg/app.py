import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise
import gymnasium as gym
from risk_analysis import calculate_var, calculate_es

# Prepare the data for LSTM
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), :]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

# Predict future stock prices for the next 5 days
def predict_future_prices(model, data, time_step, days=5):
    predictions = []
    last_data = data[-time_step:]
    for _ in range(days):
        last_data_scaled = scaler.transform(last_data)
        X_input = last_data_scaled[:, 0].reshape(1, time_step, 1)  # Use only the 'Close' feature for prediction
        pred_price = model.predict(X_input)
        predictions.append(pred_price[0][0])
        new_row = np.append(last_data[1:], [[pred_price[0][0], np.nan, np.nan, np.nan]], axis=0)
        last_data = new_row
    # Create a placeholder array with the same number of features as the original data
    placeholder = np.zeros((len(predictions), data.shape[1]))
    placeholder[:, 0] = predictions  # Fill the first column with the predicted prices
    real_prices = scaler.inverse_transform(placeholder)[:, 0]  # Only return the first column
    return real_prices

# Define the trading environment
class TradingEnv(gym.Env):
    def __init__(self, df):
        super(TradingEnv, self).__init__()
        self.df = df
        self.current_step = 0
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(df.columns),), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.current_step = 0
        return self.df.iloc[self.current_step].values, {}

    def step(self, action):
        self.current_step += 1
        if self.current_step >= len(self.df):
            self.current_step = 0
        reward = self.df.iloc[self.current_step]['Close'] * action[0]
        done = self.current_step == len(self.df) - 1
        obs = self.df.iloc[self.current_step].values
        return obs, reward, done, False, {}

# Get the list of stock symbols from the data folder
data_folder = "data"
symbols = [file.split(".")[0] for file in os.listdir(data_folder) if file.endswith(".csv")]

time_step = 100

for symbol in symbols:
    # Load and preprocess the data
    file_path = os.path.join(data_folder, f"{symbol}.csv")
    data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

    # Ensure there are no NaN values in the input data
    data.fillna(method='ffill', inplace=True)

    # Normalize the data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[['Close', 'SMA_10', 'SMA_50', 'RSI']])

    X, Y = create_dataset(data_scaled, time_step)

    # Reshape input to be [samples, time steps, features]
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2])

    # Load the LSTM model
    model_path = os.path.join('models', f'{symbol}_lstm_model.h5')
    lstm_model = load_model(model_path)

    # Predict future stock prices for the next 5 days
    future_prices = predict_future_prices(lstm_model, data[['Close', 'SMA_10', 'SMA_50', 'RSI']].values, time_step, days=5)

    # Print the predicted prices
    print(f"Predicted future stock prices for {symbol} for the next 5 days:")
    print(future_prices)

# Create the environment and load the DDPG model for backtesting (using AAPL as an example)
env = DummyVecEnv([lambda: TradingEnv(data)])
ddpg_model_path = os.path.join('models', 'ddpg_model')
ddpg_model = DDPG.load(ddpg_model_path)

# Backtest the model (using AAPL as an example)
obs = env.reset()
for _ in range(len(data)):
    action, _states = ddpg_model.predict(obs)
    obs, reward, done, _ = env.step(action)
    if done:
        break

# Calculate risk metrics (using random returns as an example)
returns = np.random.normal(0, 1, 1000)
var = calculate_var(returns)
es = calculate_es(returns)
print(f'Value at Risk (VaR): {var}')
print(f'Expected Shortfall (ES): {es}')