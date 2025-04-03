import os
import pandas as pd
import yfinance as yf

# Ensure 'data' folder exists
os.makedirs("data", exist_ok=True)

def calculate_indicators(df):
    """Calculate SMA & RSI for the dataset."""
    df["SMA_10"] = df["Close"].rolling(window=10).mean()  # 10-day Simple Moving Average
    df["SMA_50"] = df["Close"].rolling(window=50).mean()  # 50-day Simple Moving Average

    # RSI Calculation
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))  # Relative Strength Index (RSI)

    return df

def get_stock_data(symbols, start="2015-01-01", end="2025-04-02"):
    for symbol in symbols:
        stock = yf.download(symbol, start=start, end=end)
        
        # Reset index so 'Date' is a column
        stock.reset_index(inplace=True)

        # Keep only required columns
        stock = stock[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

        # Calculate SMA & RSI
        stock = calculate_indicators(stock)

        # Save the processed data
        file_path = os.path.join("data", f"{symbol}.csv")
        stock.to_csv(file_path, index=False)
        print(f"✅ Stock data for {symbol} with indicators saved at {file_path}.")

        # Display the data
        print(stock.head(20))  # Display the first 20 rows to see the indicators

# List of stock symbols
symbols = ["AMZN", "AAPL","IBM", "NFLX", "NVDA", "ORCL"]

if __name__ == "__main__":
    get_stock_data(symbols)