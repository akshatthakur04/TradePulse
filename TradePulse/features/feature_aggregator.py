"""
Feature aggregator module for merging price and sentiment data.

This module aggregates price data, technical indicators, and sentiment analysis
into a unified feature set ready for model training and inference.
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.logger import setup_logger
import config
from features.technical_indicators import TechnicalIndicators, calculate_returns

logger = setup_logger('feature_aggregator')

class FeatureAggregator:
    """
    Aggregator for merging different data sources into features.
    """
    
    def __init__(self):
        """Initialize feature aggregator."""
        self.tech_indicators = TechnicalIndicators()
    
    def merge_price_and_sentiment(self, price_df, sentiment_df, window_size=30):
        """
        Merge price data with sentiment data based on timestamps.
        
        Args:
            price_df (pd.DataFrame): Price data with OHLCV and technical indicators
            sentiment_df (pd.DataFrame): Sentiment data from news and social media
            window_size (int): Size of the lookback window in minutes
        
        Returns:
            pd.DataFrame: Merged dataframe with price and sentiment features
        """
        if price_df.empty or sentiment_df.empty:
            logger.warning("Empty dataframe received, cannot merge")
            return pd.DataFrame()
        
        # Ensure dataframes have datetime index for time operations
        price_df = price_df.copy()
        sentiment_df = sentiment_df.copy()
        
        if 'timestamp' in price_df.columns:
            price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
            price_df = price_df.set_index('timestamp')
        
        if 'timestamp' not in sentiment_df.columns:
            logger.error("Sentiment DataFrame must have a timestamp column")
            return pd.DataFrame()
        
        sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp'])
        
        # Initialize result dataframe with price data
        result_df = price_df.copy()
        
        # Group sentiment data by symbol if it exists
        if 'symbol' in sentiment_df.columns:
            sentiment_grouped = sentiment_df.groupby('symbol')
        else:
            # If no symbol, treat all sentiment as applying to all price data
            sentiment_grouped = {'global': sentiment_df}
        
        # For each symbol in price data
        for symbol in result_df['symbol'].unique() if 'symbol' in result_df.columns else ['global']:
            # Get price data for this symbol
            symbol_price = result_df[result_df['symbol'] == symbol] if 'symbol' in result_df.columns else result_df
            
            # Get sentiment data for this symbol
            if symbol in sentiment_grouped:
                symbol_sentiment = sentiment_grouped[symbol]
            elif 'global' in sentiment_grouped:
                symbol_sentiment = sentiment_grouped['global']
            else:
                # No matching sentiment, skip
                continue
            
            # For each timestamp in the price data
            for idx, timestamp in enumerate(symbol_price.index):
                # Get sentiment data from the lookback window
                window_start = timestamp - timedelta(minutes=window_size)
                
                # Find sentiment within the window
                window_sentiment = symbol_sentiment[
                    (symbol_sentiment['timestamp'] >= window_start) & 
                    (symbol_sentiment['timestamp'] <= timestamp)
                ]
                
                if not window_sentiment.empty:
                    # Aggregate sentiment metrics in the window
                    # VADER sentiment
                    if 'vader_compound' in window_sentiment.columns:
                        result_df.at[timestamp, 'vader_compound_mean'] = window_sentiment['vader_compound'].mean()
                        result_df.at[timestamp, 'vader_compound_std'] = window_sentiment['vader_compound'].std()
                        result_df.at[timestamp, 'vader_compound_min'] = window_sentiment['vader_compound'].min()
                        result_df.at[timestamp, 'vader_compound_max'] = window_sentiment['vader_compound'].max()
                        
                        # Positive/negative counts
                        pos_count = (window_sentiment['vader_compound'] > config.VADER_THRESHOLD_POSITIVE).sum()
                        neg_count = (window_sentiment['vader_compound'] < config.VADER_THRESHOLD_NEGATIVE).sum()
                        result_df.at[timestamp, 'vader_positive_count'] = pos_count
                        result_df.at[timestamp, 'vader_negative_count'] = neg_count
                        result_df.at[timestamp, 'vader_ratio'] = pos_count / (neg_count + 1)  # Add 1 to avoid div by zero
                    
                    # FinBERT sentiment
                    if 'finbert_compound' in window_sentiment.columns:
                        result_df.at[timestamp, 'finbert_compound_mean'] = window_sentiment['finbert_compound'].mean()
                        result_df.at[timestamp, 'finbert_compound_std'] = window_sentiment['finbert_compound'].std()
                        
                        # Positive/negative counts
                        pos_count = (window_sentiment['finbert_sentiment'] == 'positive').sum()
                        neg_count = (window_sentiment['finbert_sentiment'] == 'negative').sum()
                        result_df.at[timestamp, 'finbert_positive_count'] = pos_count
                        result_df.at[timestamp, 'finbert_negative_count'] = neg_count
                        result_df.at[timestamp, 'finbert_ratio'] = pos_count / (neg_count + 1)
                    
                    # Source counting
                    if 'source' in window_sentiment.columns:
                        # Count news sources vs social media
                        news_count = window_sentiment[window_sentiment['source'].isin(
                            ['reuters_business', 'reuters_markets', 'seeking_alpha_market_news', 'yahoo_finance', 
                             'cnbc_investing', 'investing_com_news'])].shape[0]
                        
                        social_count = window_sentiment[window_sentiment['source'].isin(
                            ['stocktwits', 'reddit'])].shape[0]
                        
                        result_df.at[timestamp, 'news_count'] = news_count
                        result_df.at[timestamp, 'social_count'] = social_count
                        result_df.at[timestamp, 'total_mentions'] = window_sentiment.shape[0]
                
                else:
                    # No sentiment in window, set to neutral
                    result_df.at[timestamp, 'vader_compound_mean'] = 0
                    result_df.at[timestamp, 'vader_compound_std'] = 0
                    result_df.at[timestamp, 'vader_positive_count'] = 0
                    result_df.at[timestamp, 'vader_negative_count'] = 0
                    result_df.at[timestamp, 'vader_ratio'] = 1
                    
                    if 'finbert_compound' in window_sentiment.columns:
                        result_df.at[timestamp, 'finbert_compound_mean'] = 0
                        result_df.at[timestamp, 'finbert_compound_std'] = 0
                        result_df.at[timestamp, 'finbert_positive_count'] = 0
                        result_df.at[timestamp, 'finbert_negative_count'] = 0
                        result_df.at[timestamp, 'finbert_ratio'] = 1
                    
                    result_df.at[timestamp, 'news_count'] = 0
                    result_df.at[timestamp, 'social_count'] = 0
                    result_df.at[timestamp, 'total_mentions'] = 0
        
        # Fill NaN values from the sentiment merging
        sentiment_cols = [col for col in result_df.columns if col.startswith(('vader_', 'finbert_', 'news_', 'social_', 'total_'))]
        result_df[sentiment_cols] = result_df[sentiment_cols].fillna(0)
        
        return result_df
    
    def prepare_complete_features(self, price_df, sentiment_df=None, include_returns=True):
        """
        Prepare a complete feature set with technical indicators and sentiment.
        
        Args:
            price_df (pd.DataFrame): Price data with OHLCV
            sentiment_df (pd.DataFrame, optional): Sentiment data
            include_returns (bool): Whether to include future returns (for training)
            
        Returns:
            pd.DataFrame: Complete feature dataframe
        """
        if price_df.empty:
            logger.error("Empty price dataframe, cannot prepare features")
            return pd.DataFrame()
        
        # Add technical indicators
        features_df = self.tech_indicators.add_all_indicators(price_df)
        
        # Add sentiment features if provided
        if sentiment_df is not None and not sentiment_df.empty:
            features_df = self.merge_price_and_sentiment(features_df, sentiment_df)
        
        # Add returns if needed (for training)
        if include_returns:
            features_df = calculate_returns(features_df)
        
        # Drop rows with NaN values (from indicators or lookbacks)
        features_df = features_df.dropna()
        
        return features_df

def prepare_model_features(features_df, target_col='return_1', drop_cols=None):
    """
    Prepare features for model training or inference.
    
    Args:
        features_df (pd.DataFrame): DataFrame with all features
        target_col (str): Name of target column for prediction
        drop_cols (list): List of columns to drop from features
        
    Returns:
        tuple: X (features), y (target if target_col exists, otherwise None)
    """
    if features_df.empty:
        logger.error("Empty features dataframe")
        return pd.DataFrame(), None
    
    # Make a copy to avoid modifying the input
    model_df = features_df.copy()
    
    # Columns to drop from features
    if drop_cols is None:
        drop_cols = ['symbol', 'open', 'high', 'low', 'close', 'volume']
    
    # Drop non-feature columns that exist in the dataframe
    feature_cols = [col for col in model_df.columns if col not in drop_cols and col != target_col]
    
    # Extract features
    X = model_df[feature_cols]
    
    # Extract target if it exists
    y = None
    if target_col in model_df.columns:
        y = model_df[target_col]
    
    return X, y

if __name__ == "__main__":
    # Example usage
    import numpy as np
    from datetime import datetime, timedelta
    
    # Create sample price data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='1min')
    np.random.seed(42)
    
    # Random walk price series
    closes = [100]
    for _ in range(1, 100):
        closes.append(closes[-1] * (1 + np.random.normal(0, 0.001)))
    
    price_df = pd.DataFrame({
        'timestamp': dates,
        'symbol': 'EXAMPLE',
        'open': closes,
        'high': [c * 1.001 for c in closes],
        'low': [c * 0.999 for c in closes],
        'close': closes,
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    # Create sample sentiment data
    sentiment_dates = []
    sentiments = []
    sources = []
    for i in range(20):
        # Random timestamp within the price range
        random_idx = np.random.randint(0, 100)
        sentiment_dates.append(dates[random_idx])
        # Random sentiment between -1 and 1
        sentiments.append(np.random.uniform(-1, 1))
        # Random source (news or social)
        sources.append(np.random.choice(['reuters_business', 'stocktwits']))
    
    sentiment_df = pd.DataFrame({
        'timestamp': sentiment_dates,
        'symbol': 'EXAMPLE',
        'vader_compound': sentiments,
        'finbert_sentiment': ['positive' if s > 0 else 'negative' for s in sentiments],
        'finbert_compound': sentiments,
        'source': sources
    })
    
    # Create feature aggregator
    aggregator = FeatureAggregator()
    
    # Add indicators to price data
    price_with_indicators = aggregator.tech_indicators.add_all_indicators(price_df)
    
    # Merge price and sentiment
    merged_df = aggregator.merge_price_and_sentiment(price_with_indicators, sentiment_df)
    
    # Prepare complete feature set
    complete_features = aggregator.prepare_complete_features(price_df, sentiment_df)
    
    # Prepare model features
    X, y = prepare_model_features(complete_features)
    
    # Print results
    print(f"Price data shape: {price_df.shape}")
    print(f"Sentiment data shape: {sentiment_df.shape}")
    print(f"Price with indicators shape: {price_with_indicators.shape}")
    print(f"Merged data shape: {merged_df.shape}")
    print(f"Complete features shape: {complete_features.shape}")
    print(f"Model features (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape if y is not None else None}")
    
    # Show some sentiment features
    sentiment_cols = [col for col in merged_df.columns if col.startswith(('vader_', 'finbert_'))]
    print("\nSample sentiment features:")
    if sentiment_cols:
        print(merged_df[sentiment_cols].describe().round(3)) 