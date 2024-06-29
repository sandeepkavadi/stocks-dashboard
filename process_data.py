import pandas as pd
import numpy as np
import yfinance as yf
import os
import datetime as dt

df_portfolio = pd.read_csv('./data/portfolio.csv')
tickers = df_portfolio['Ticker'].tolist()

# Initialize: First run
def get_initial_data(tickers: list, days:int) -> pd.DataFrame:
    """Get initial data for the last n days.
    Args:
    tickers (list): List of tickers
    days (int): Number of days to get data for
    Usage:
    get_initial_data(['AAPL', 'GOOGL'], 365) # Get data for the last year
    """
    print(f'Getting initial data for the last {days} days')
    start_date = dt.datetime.today() - dt.timedelta(days=days)
    end_date = dt.datetime.today()  
    prices = yf.download(tickers, start=start_date, end=end_date)['Adj Close']  
    prices.to_parquet('./data/prices.parquet')
    return prices

# Read in data
def read_prices() -> pd.DataFrame:
    """Read in the prices data.
    Returns:
    prices (pd.DataFrame): Prices data
    Usage:
    prices = read_prices()
    """
    prices = pd.read_parquet('./data/prices.parquet')
    return prices

# Get incremental data
def get_incremental_data(tickers: list, latest_available_date: dt.datetime) -> pd.DataFrame:
    """Get incremental data since the latest available date.
    Args:
    tickers (list): List of tickers
    latest_available_date (dt.datetime): Latest available date
    Returns:
    prices_incremental (pd.DataFrame): Incremental prices data
    Usage:
    prices_incremental = get_incremental_data(['AAPL', 'GOOGL'], dt.datetime(2021, 1, 1))
    """
    prices_incremental = yf.download(tickers, start=latest_available_date, end=dt.datetime.today())['Adj Close']
    return prices_incremental

# Update the prices
def update_prices(prices: pd.DataFrame, prices_incremental: pd.DataFrame) -> pd.DataFrame:
    """Update the prices data with the incremental data.
    Args:
    prices (pd.DataFrame): Prices data
    prices_incremental (pd.DataFrame): Incremental prices data
    Returns:
    prices (pd.DataFrame): Updated prices data
    Usage:
    prices = update_prices(prices, prices_incremental)
    """
    prices = pd.concat([prices, prices_incremental], axis=0)
    prices = prices.drop_duplicates(keep='last')
    prices.to_parquet('./data/prices.parquet')
    return prices

# Compute daily returns
def compute_daily_returns(prices: pd.Series) -> pd.Series:
    """Compute the daily returns from the prices data.
    Args:
    prices (pd.Series): Prices data
    Returns:
    returns (pd.Series): Returns data
    Usage:
    returns = compute_daily_returns(prices)
    """
    # daily returns
    returns = prices.dropna().pct_change()
    return returns

# Compute weekly returns
def compute_weekly_returns(prices: pd.Series) -> pd.Series:
    """Compute the weekly returns from the prices data.
    Args:
    prices (pd.Series): Prices data
    Returns:
    returns (pd.Series): Returns data
    Usage:
    returns = compute_weekly_returns(prices)
    """
    # weekly returns
    returns = prices.dropna().resample('W').ffill().pct_change()
    return returns


# Create the returns dataframe
def get_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Create the returns dataframe.
    Args:
    prices (pd.DataFrame): Prices data
    Returns:
    returns (pd.DataFrame): Returns data
    Usage:
    returns = get_returns(prices)
    """
    daily_returns = []
    weekly_returns = []
    for col in prices.columns:
        daily_returns.append(compute_daily_returns(prices[col]))
        weekly_returns.append(compute_weekly_returns(prices[col]))
    daily_returns = pd.concat(daily_returns, axis=1).dropna(how='all', axis=0)
    weekly_returns = pd.concat(weekly_returns, axis=1).dropna(how='all', axis=0)
    # write daily returns to parquet
    daily_returns.to_parquet('./data/daily_returns.parquet')
    weekly_returns.to_parquet('./data/weekly_returns.parquet')
    return 


# Main function
def main():
    # check if the data folder exists
    if not os.path.exists('./data'):
        os.makedirs('./data')
    # check if prices.parquet exists
    if not os.path.exists('./data/prices.parquet'):
        print('Initial data not found. Getting initial data...')
        prices = get_initial_data(tickers, 5*365) # get initial data for the last 5 years
    else:
        print('Initial data found. Reading data...')
        prices = read_prices()
    print(f'Successfuly got prices for the tickers. Length of prices: {len(prices)}')
    latest_date = prices.index.max()
    print(f'Latest available date is: {latest_date}. Today is: {dt.datetime.today()}')
    if latest_date < dt.datetime.today() - dt.timedelta(days=2):
        print('Data is stale. Updating prices...')
        prices_incremental = get_incremental_data(tickers, latest_date)
        prices = update_prices(prices, prices_incremental)
    # compute and write daily returns
    get_returns(prices)


if __name__ == '__main__':
    main()
