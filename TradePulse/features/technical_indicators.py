"""
Technical indicators module for financial time series.

This module provides functions to calculate common technical indicators
like SMA, EMA, RSI, MACD, and Bollinger Bands using the TA library.
"""
import os
import sys
import pandas as pd
import numpy as np
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.logger import setup_logger
import config

logger = setup_logger('technical_indicators')

class TechnicalIndicators:
    """Technical indicators calculator for price dataframes."""
    
    def __init__(self, df=None):
        """
        Initialize with optional dataframe.
        
        Args:
            df (pd.DataFrame, optional): DataFrame with OHLCV data
        """
        self.df = df
    
    def add_sma(self, df=None, periods=None, close_col='close', add_distance=True):
        """
        Add Simple Moving Average indicators.
        
        Args:
            df (pd.DataFrame, optional): DataFrame to use (uses self.df if None)
            periods (list, optional): List of periods to calculate SMA for
            close_col (str): Column name for close prices
            add_distance (bool): Whether to add distance from price to SMA
            
        Returns:
            pd.DataFrame: DataFrame with added SMA columns
        """
        df = df if df is not None else self.df
        if df is None or df.empty:
            logger.error("No dataframe provided for SMA calculation")
            return pd.DataFrame()
            
        periods = periods or config.SMA_PERIODS
        
        for period in periods:
            column_name = f'sma_{period}'
            
            # Calculate SMA
            sma_indicator = SMAIndicator(close=df[close_col], window=period)
            df[column_name] = sma_indicator.sma_indicator()
            
            # Calculate distance from price (percent)
            if add_distance:
                df[f'{column_name}_dist'] = (df[close_col] - df[column_name]) / df[column_name] * 100
        
        return df
    
    def add_ema(self, df=None, periods=None, close_col='close', add_distance=True):
        """
        Add Exponential Moving Average indicators.
        
        Args:
            df (pd.DataFrame, optional): DataFrame to use (uses self.df if None)
            periods (list, optional): List of periods to calculate EMA for
            close_col (str): Column name for close prices
            add_distance (bool): Whether to add distance from price to EMA
            
        Returns:
            pd.DataFrame: DataFrame with added EMA columns
        """
        df = df if df is not None else self.df
        if df is None or df.empty:
            logger.error("No dataframe provided for EMA calculation")
            return pd.DataFrame()
            
        periods = periods or config.EMA_PERIODS
        
        for period in periods:
            column_name = f'ema_{period}'
            
            # Calculate EMA
            ema_indicator = EMAIndicator(close=df[close_col], window=period)
            df[column_name] = ema_indicator.ema_indicator()
            
            # Calculate distance from price (percent)
            if add_distance:
                df[f'{column_name}_dist'] = (df[close_col] - df[column_name]) / df[column_name] * 100
        
        return df
    
    def add_rsi(self, df=None, period=None, close_col='close'):
        """
        Add Relative Strength Index indicator.
        
        Args:
            df (pd.DataFrame, optional): DataFrame to use (uses self.df if None)
            period (int, optional): Period for RSI calculation
            close_col (str): Column name for close prices
            
        Returns:
            pd.DataFrame: DataFrame with added RSI column
        """
        df = df if df is not None else self.df
        if df is None or df.empty:
            logger.error("No dataframe provided for RSI calculation")
            return pd.DataFrame()
            
        period = period or config.RSI_PERIOD
        
        # Calculate RSI
        rsi_indicator = RSIIndicator(close=df[close_col], window=period)
        df[f'rsi_{period}'] = rsi_indicator.rsi()
        
        return df
    
    def add_macd(self, df=None, close_col='close', 
                 fast=None, slow=None, signal=None):
        """
        Add Moving Average Convergence Divergence indicator.
        
        Args:
            df (pd.DataFrame, optional): DataFrame to use (uses self.df if None)
            close_col (str): Column name for close prices
            fast (int, optional): Fast period for MACD calculation
            slow (int, optional): Slow period for MACD calculation
            signal (int, optional): Signal period for MACD calculation
            
        Returns:
            pd.DataFrame: DataFrame with added MACD columns
        """
        df = df if df is not None else self.df
        if df is None or df.empty:
            logger.error("No dataframe provided for MACD calculation")
            return pd.DataFrame()
            
        fast = fast or config.MACD_FAST
        slow = slow or config.MACD_SLOW
        signal = signal or config.MACD_SIGNAL
        
        # Calculate MACD
        macd_indicator = MACD(
            close=df[close_col],
            window_slow=slow,
            window_fast=fast,
            window_sign=signal
        )
        
        # Add MACD line, signal, and histogram
        df[f'macd_line'] = macd_indicator.macd()
        df[f'macd_signal'] = macd_indicator.macd_signal()
        df[f'macd_hist'] = macd_indicator.macd_diff()
        
        return df
    
    def add_bollinger_bands(self, df=None, period=None, std_dev=2, close_col='close'):
        """
        Add Bollinger Bands indicator.
        
        Args:
            df (pd.DataFrame, optional): DataFrame to use (uses self.df if None)
            period (int, optional): Period for Bollinger Bands calculation
            std_dev (float): Standard deviation factor for bands
            close_col (str): Column name for close prices
            
        Returns:
            pd.DataFrame: DataFrame with added Bollinger Bands columns
        """
        df = df if df is not None else self.df
        if df is None or df.empty:
            logger.error("No dataframe provided for Bollinger Bands calculation")
            return pd.DataFrame()
            
        period = period or config.BOLLINGER_PERIOD
        
        # Calculate Bollinger Bands
        bb_indicator = BollingerBands(
            close=df[close_col],
            window=period,
            window_dev=std_dev
        )
        
        # Add bands and width
        df[f'bb_{period}_upper'] = bb_indicator.bollinger_hband()
        df[f'bb_{period}_middle'] = bb_indicator.bollinger_mavg()
        df[f'bb_{period}_lower'] = bb_indicator.bollinger_lband()
        df[f'bb_{period}_width'] = bb_indicator.bollinger_wband()
        
        # Add percent B (position within bands)
        df[f'bb_{period}_pctb'] = (df[close_col] - df[f'bb_{period}_lower']) / (df[f'bb_{period}_upper'] - df[f'bb_{period}_lower'])
        
        return df
    
    def add_all_indicators(self, df=None, close_col='close'):
        """
        Add all technical indicators to the dataframe.
        
        Args:
            df (pd.DataFrame, optional): DataFrame to use (uses self.df if None)
            close_col (str): Column name for close prices
            
        Returns:
            pd.DataFrame: DataFrame with all technical indicators added
        """
        df = df if df is not None else self.df
        if df is None or df.empty:
            logger.error("No dataframe provided for indicator calculations")
            return pd.DataFrame()
        
        # Create a copy to avoid modifying the input dataframe
        df_with_indicators = df.copy()
        
        # Add all indicators
        df_with_indicators = self.add_sma(df_with_indicators, close_col=close_col)
        df_with_indicators = self.add_ema(df_with_indicators, close_col=close_col)
        df_with_indicators = self.add_rsi(df_with_indicators, close_col=close_col)
        df_with_indicators = self.add_macd(df_with_indicators, close_col=close_col)
        df_with_indicators = self.add_bollinger_bands(df_with_indicators, close_col=close_col)
        
        return df_with_indicators

def calculate_returns(df, close_col='close', periods=[1, 5, 10]):
    """
    Calculate future returns for different periods.
    
    Args:
        df (pd.DataFrame): DataFrame with price data
        close_col (str): Column name for close prices
        periods (list): List of periods to calculate returns for
        
    Returns:
        pd.DataFrame: DataFrame with added return columns
    """
    if df is None or df.empty or close_col not in df.columns:
        logger.error("Invalid dataframe provided for returns calculation")
        return pd.DataFrame()
    
    result_df = df.copy()
    
    for period in periods:
        # Calculate future return (this will have NaN values at the end)
        result_df[f'return_{period}'] = result_df[close_col].pct_change(period).shift(-period) * 100
    
    return result_df

def prepare_features(price_df, include_returns=True):
    """
    Prepare features dataframe with technical indicators.
    
    Args:
        price_df (pd.DataFrame): DataFrame with OHLCV price data
        include_returns (bool): Whether to include future returns (for training)
        
    Returns:
        pd.DataFrame: DataFrame with features
    """
    if price_df is None or price_df.empty:
        logger.error("No price data provided for feature preparation")
        return pd.DataFrame()
    
    # Create technical indicators
    indicators = TechnicalIndicators()
    features_df = indicators.add_all_indicators(price_df)
    
    # Add returns if requested (for training/backtesting)
    if include_returns:
        features_df = calculate_returns(features_df)
    
    # Drop rows with NaN (from indicator calculations)
    features_df = features_df.dropna()
    
    return features_df

if __name__ == "__main__":
    # Example usage
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Create sample OHLCV data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='1D')
    np.random.seed(42)
    
    # Random walk price series
    closes = [100]
    for _ in range(1, 100):
        closes.append(closes[-1] * (1 + np.random.normal(0, 0.02)))
    
    high = [c * (1 + abs(np.random.normal(0, 0.01))) for c in closes]
    low = [c * (1 - abs(np.random.normal(0, 0.01))) for c in closes]
    
    df = pd.DataFrame({
        'timestamp': dates,
        'symbol': 'EXAMPLE',
        'open': closes,
        'high': high,
        'low': low,
        'close': closes,
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    # Calculate indicators
    indicators = TechnicalIndicators()
    result = indicators.add_all_indicators(df)
    
    # Add returns
    result = calculate_returns(result)
    
    # Print result
    print(f"Generated features dataframe with {result.shape[1]} columns")
    print("\nSample of generated features:")
    print(result[['close', 'sma_5', 'ema_10', 'rsi_14', 'macd_line', 'bb_20_width', 'return_1']].tail().round(2)) 