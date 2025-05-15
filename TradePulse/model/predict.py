"""
Model prediction module for stock price prediction.

This module provides functionality to load trained models and make predictions
on new market data for real-time trading signals.
"""
import os
import sys
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.logger import setup_logger
import config
from features.feature_aggregator import FeatureAggregator, prepare_model_features
from features.technical_indicators import prepare_features
from model.train import LightGBMModel, ARIMAXModel

logger = setup_logger('model_predict')

def load_latest_model(model_dir='models', model_type='lightgbm'):
    """
    Load the most recently trained model of the specified type.
    
    Args:
        model_dir (str): Directory containing model files
        model_type (str): Type of model to load ('lightgbm' or 'arimax')
        
    Returns:
        object: Loaded model or None if not found
    """
    if not os.path.exists(model_dir):
        logger.error(f"Model directory {model_dir} does not exist")
        return None
    
    # Find model files of the specified type
    if model_type.lower() == 'lightgbm':
        model_files = [f for f in os.listdir(model_dir) if f.startswith('lightgbm_') and f.endswith('.txt')]
    elif model_type.lower() == 'arimax':
        model_files = [f for f in os.listdir(model_dir) if f.startswith('arimax_') and f.endswith('.pkl')]
    else:
        logger.error(f"Unsupported model type: {model_type}")
        return None
    
    if not model_files:
        logger.error(f"No {model_type} models found in {model_dir}")
        return None
    
    # Sort by timestamp in filename (assuming format model_YYYYMMDD_HHMMSS.ext)
    model_files.sort(reverse=True)
    latest_file = model_files[0]
    
    # Load the model
    try:
        model_path = os.path.join(model_dir, latest_file)
        logger.info(f"Loading model from {model_path}")
        
        if model_type.lower() == 'lightgbm':
            model = LightGBMModel.load_model(model_path)
        else:  # arimax
            model = ARIMAXModel.load_model(model_path)
        
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

def load_model(model_path):
    """
    Load a model from a specific path.
    
    Args:
        model_path (str): Path to the model file
        
    Returns:
        object: Loaded model or None if error
    """
    if not os.path.exists(model_path):
        logger.error(f"Model file {model_path} does not exist")
        return None
    
    try:
        # Determine model type from file extension
        if model_path.endswith('.txt'):
            logger.info(f"Loading LightGBM model from {model_path}")
            model = LightGBMModel.load_model(model_path)
        elif model_path.endswith('.pkl'):
            logger.info(f"Loading ARIMAX model from {model_path}")
            model = ARIMAXModel.load_model(model_path)
        else:
            logger.error(f"Unknown model file type: {model_path}")
            return None
        
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

def prepare_data_for_prediction(price_df, sentiment_df=None):
    """
    Prepare raw data for prediction by creating features.
    
    Args:
        price_df (pd.DataFrame): Dataframe with OHLCV price data
        sentiment_df (pd.DataFrame, optional): Dataframe with sentiment data
        
    Returns:
        pd.DataFrame: Features ready for model prediction
    """
    if price_df.empty:
        logger.error("Empty price dataframe provided")
        return pd.DataFrame()
    
    try:
        # Create feature aggregator
        aggregator = FeatureAggregator()
        
        # Prepare complete features (without returns for prediction)
        features_df = aggregator.prepare_complete_features(price_df, sentiment_df, include_returns=False)
        
        # Drop rows with NaN values
        features_df = features_df.dropna()
        
        return features_df
        
    except Exception as e:
        logger.error(f"Error preparing data for prediction: {e}")
        return pd.DataFrame()

def predict_returns(model, features_df):
    """
    Generate return predictions using a trained model.
    
    Args:
        model: Trained LightGBM or ARIMAX model
        features_df (pd.DataFrame): Features dataframe
        
    Returns:
        pd.Series or np.array: Predicted returns
    """
    if features_df.empty:
        logger.error("Empty features dataframe")
        return None
    
    if model is None:
        logger.error("No model provided for prediction")
        return None
    
    try:
        # Get model type
        model_type = model.__class__.__name__
        
        if model_type == 'LightGBMModel':
            # Prepare features for LightGBM
            X, _ = prepare_model_features(features_df, target_col=None)
            
            # Make predictions
            predictions = model.predict(X)
            
            # Create a Series with DateTimeIndex
            if 'timestamp' in features_df.columns:
                index = features_df['timestamp']
            else:
                index = features_df.index
            
            return pd.Series(predictions, index=index, name='predicted_return')
            
        elif model_type == 'ARIMAXModel':
            # For ARIMAX, we need to extract the required exog variables
            if hasattr(model, 'exog_columns') and model.exog_columns is not None:
                # Filter to only required columns
                exog = features_df[model.exog_columns]
                
                # Make predictions (one step ahead)
                predictions = model.predict(steps=1, exog=exog)
                
                return predictions
            else:
                # No exogenous variables needed
                predictions = model.predict(steps=1)
                
                return predictions
        else:
            logger.error(f"Unsupported model type: {model_type}")
            return None
            
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        return None

def generate_trading_signals(predicted_returns, threshold=0.1):
    """
    Generate trading signals based on predicted returns.
    
    Args:
        predicted_returns (pd.Series): Series of predicted returns
        threshold (float): Return threshold to generate buy/sell signal
        
    Returns:
        pd.DataFrame: DataFrame with trading signals
    """
    if predicted_returns is None or len(predicted_returns) == 0:
        logger.error("No predictions provided")
        return pd.DataFrame()
    
    # Create signal DataFrame
    signals = pd.DataFrame()
    
    if isinstance(predicted_returns, pd.Series):
        signals['timestamp'] = predicted_returns.index
        signals['predicted_return'] = predicted_returns.values
    else:
        # If it's just an array or list
        signals['predicted_return'] = predicted_returns
        signals['timestamp'] = pd.date_range(
            start=datetime.now(), 
            periods=len(predicted_returns), 
            freq='1min'
        )
    
    # Generate signals based on predicted returns
    signals['signal'] = 'HOLD'
    signals.loc[signals['predicted_return'] > threshold, 'signal'] = 'BUY'
    signals.loc[signals['predicted_return'] < -threshold, 'signal'] = 'SELL'
    
    # Add confidence score (simple linear scaling based on return magnitude)
    signals['confidence'] = signals['predicted_return'].abs() / signals['predicted_return'].abs().max()
    signals['confidence'] = signals['confidence'].fillna(0)
    
    return signals

def predict_next_minute(model, current_data, sentiment_data=None):
    """
    Generate prediction for the next minute.
    
    Args:
        model: Trained model (LightGBM or ARIMAX)
        current_data (pd.DataFrame): Current price data
        sentiment_data (pd.DataFrame, optional): Current sentiment data
        
    Returns:
        dict: Prediction results
    """
    # Prepare features
    features = prepare_data_for_prediction(current_data, sentiment_data)
    
    if features.empty:
        logger.error("Failed to generate features for prediction")
        return {
            'error': True,
            'message': 'Failed to generate features',
            'timestamp': datetime.now(),
            'signal': 'HOLD',
            'predicted_return': 0.0,
            'confidence': 0.0
        }
    
    # Generate prediction
    predicted_returns = predict_returns(model, features)
    
    if predicted_returns is None or len(predicted_returns) == 0:
        logger.error("Failed to generate prediction")
        return {
            'error': True,
            'message': 'Failed to generate prediction',
            'timestamp': datetime.now(),
            'signal': 'HOLD',
            'predicted_return': 0.0,
            'confidence': 0.0
        }
    
    # Get the last prediction (should be only one for next minute)
    last_return = predicted_returns[-1] if isinstance(predicted_returns, (list, np.ndarray)) else predicted_returns.iloc[-1]
    
    # Generate signal
    signal = 'HOLD'
    if last_return > 0.1:  # Threshold for buy
        signal = 'BUY'
    elif last_return < -0.1:  # Threshold for sell
        signal = 'SELL'
    
    # Calculate confidence (simple scaling)
    confidence = min(abs(last_return) * 10, 1.0)
    
    # Return prediction result
    return {
        'error': False,
        'timestamp': datetime.now(),
        'symbol': current_data['symbol'].iloc[-1] if 'symbol' in current_data.columns else 'UNKNOWN',
        'predicted_return': float(last_return),
        'signal': signal,
        'confidence': float(confidence)
    }

def predict_multiple_symbols(model, price_dict, sentiment_df=None):
    """
    Generate predictions for multiple symbols.
    
    Args:
        model: Trained model (LightGBM or ARIMAX)
        price_dict (dict): Dictionary mapping symbols to price DataFrames
        sentiment_df (pd.DataFrame, optional): Sentiment data for all symbols
        
    Returns:
        dict: Dictionary mapping symbols to prediction results
    """
    results = {}
    
    for symbol, price_df in price_dict.items():
        # Filter sentiment data for this symbol if available
        symbol_sentiment = None
        if sentiment_df is not None and not sentiment_df.empty and 'symbol' in sentiment_df.columns:
            symbol_sentiment = sentiment_df[sentiment_df['symbol'] == symbol]
        
        # Generate prediction
        prediction = predict_next_minute(model, price_df, symbol_sentiment)
        results[symbol] = prediction
    
    return results

if __name__ == "__main__":
    # Example usage
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Create sample price data
    dates = pd.date_range(end=datetime.now(), periods=100, freq='1min')
    np.random.seed(42)
    
    # Random walk price series
    closes = [100]
    for _ in range(1, 100):
        closes.append(closes[-1] * (1 + np.random.normal(0, 0.001)))
    
    price_df = pd.DataFrame({
        'timestamp': dates,
        'symbol': 'EXAMPLE',
        'open': closes,
        'high': [c * (1 + abs(np.random.normal(0, 0.001))) for c in closes],
        'low': [c * (1 - abs(np.random.normal(0, 0.001))) for c in closes],
        'close': closes,
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    # Check if a trained model exists
    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created model directory: {model_dir}")
        print("No existing models found. First run train.py to train a model.")
        
        # Create a simple dummy model for demonstration
        from model.train import LightGBMModel
        import lightgbm as lgb
        
        # Add technical indicators to price data for features
        features = prepare_features(price_df)
        
        # Add target (future returns)
        features['return_1'] = features['close'].pct_change(-1) * 100
        features = features.dropna()
        
        # Prepare model data
        X, y = prepare_model_features(features)
        
        # Create and fit a simple model
        model = LightGBMModel()
        model.fit(X, y)
        
        # Save model
        model_path = os.path.join(model_dir, 'lightgbm_example.txt')
        model.save_model(model_path)
        print(f"Created example model at {model_path}")
    else:
        # Load the latest model
        model = load_latest_model(model_dir)
        if model is None:
            print("No existing models found. First run train.py to train a model.")
            exit(1)
    
    # Generate features for prediction
    features_df = prepare_data_for_prediction(price_df)
    print(f"Generated features with shape: {features_df.shape}")
    
    # Make predictions
    predictions = predict_returns(model, features_df)
    print("\nPredicted returns:")
    print(predictions.tail())
    
    # Generate trading signals
    signals = generate_trading_signals(predictions)
    print("\nTrading signals:")
    print(signals.tail())
    
    # Next minute prediction
    next_prediction = predict_next_minute(model, price_df)
    print("\nNext minute prediction:")
    print(f"Symbol: {next_prediction['symbol']}")
    print(f"Predicted return: {next_prediction['predicted_return']:.4f}%")
    print(f"Signal: {next_prediction['signal']}")
    print(f"Confidence: {next_prediction['confidence']:.2f}")
    
    # Multiple symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    multi_price_dict = {}
    
    for symbol in symbols:
        # Clone the price data with different symbol
        symbol_df = price_df.copy()
        symbol_df['symbol'] = symbol
        
        # Add some random variation
        factor = np.random.uniform(0.98, 1.02)
        for col in ['open', 'high', 'low', 'close']:
            symbol_df[col] = symbol_df[col] * factor
        
        multi_price_dict[symbol] = symbol_df
    
    # Predict for multiple symbols
    multi_results = predict_multiple_symbols(model, multi_price_dict)
    print("\nMulti-symbol predictions:")
    for symbol, result in multi_results.items():
        print(f"{symbol}: {result['signal']} (Return: {result['predicted_return']:.4f}%)") 