# app.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise
import gymnasium as gym
import pandas as pd
from risk_analysis import calculate_var, calculate_es

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

# Load the data
data = pd.read_csv('data/AAPL.csv', parse_dates=['Date'], index_col='Date')

# Create the environment
env = DummyVecEnv([lambda: TradingEnv(data)])

# Define the model
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
model = DDPG('MlpPolicy', env, action_noise=action_noise, verbose=1)

# Train the model
model.learn(total_timesteps=10000)

# Save the model
model.save('ddpg_model')
print("✅ DDPG model trained and saved.")

# Calculate risk metrics
returns = np.random.normal(0, 1, 1000)
var = calculate_var(returns)
es = calculate_es(returns)
print(f'Value at Risk (VaR): {var}')
print(f'Expected Shortfall (ES): {es}')
