"""
Model training module for stock prediction models.

This module provides functions to train time series models for stock prediction,
including statsmodels ARIMAX and LightGBM regression models.
"""
import os
import sys
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.logger import setup_logger
import config
from features.technical_indicators import prepare_features
from features.feature_aggregator import prepare_model_features

logger = setup_logger('model_train')

class ARIMAXModel:
    """ARIMAX time series model for stock predictions."""
    
    def __init__(self, order=(1, 0, 1), seasonal_order=(0, 0, 0, 0)):
        """
        Initialize ARIMAX model with given parameters.
        
        Args:
            order (tuple): ARIMA order (p, d, q)
            seasonal_order (tuple): Seasonal order (P, D, Q, s)
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.model_fit = None
        self.exog_columns = None
        self.metadata = {
            'model_type': 'ARIMAX',
            'order': order,
            'seasonal_order': seasonal_order,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def fit(self, endog, exog=None):
        """
        Fit ARIMAX model.
        
        Args:
            endog (pd.Series): Target variable (e.g., returns)
            exog (pd.DataFrame): Exogenous variables (features)
            
        Returns:
            self: Fitted model
        """
        logger.info(f"Fitting ARIMAX model with order {self.order}")
        
        try:
            # Store exogenous column names for prediction
            if exog is not None:
                self.exog_columns = exog.columns.tolist()
                self.metadata['exog_columns'] = self.exog_columns
            
            # Create and fit model
            self.model = SARIMAX(
                endog=endog, 
                exog=exog,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            self.model_fit = self.model.fit(disp=False)
            logger.info("ARIMAX model fitting complete")
            
            # Store model metadata
            self.metadata['aic'] = self.model_fit.aic
            self.metadata['bic'] = self.model_fit.bic
            
            return self
            
        except Exception as e:
            logger.error(f"Error fitting ARIMAX model: {e}")
            raise
    
    def predict(self, steps=1, exog=None):
        """
        Make predictions with fitted ARIMAX model.
        
        Args:
            steps (int): Number of steps to forecast
            exog (pd.DataFrame): Exogenous variables for forecast period
            
        Returns:
            pd.Series: Predicted values
        """
        if self.model_fit is None:
            logger.error("Model not fitted. Call fit() before predict()")
            return None
        
        try:
            # Ensure exogenous variables match what was used for training
            if exog is not None and self.exog_columns is not None:
                exog = exog[self.exog_columns]
            
            # Make forecast
            forecast = self.model_fit.forecast(steps=steps, exog=exog)
            return forecast
            
        except Exception as e:
            logger.error(f"Error making ARIMAX prediction: {e}")
            return None
    
    def save_model(self, filepath):
        """
        Save fitted model to disk.
        
        Args:
            filepath (str): Path to save the model
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.model_fit is None:
            logger.error("Model not fitted. Call fit() before saving")
            return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save model using joblib
            joblib.dump(self, filepath)
            
            # Save metadata separately
            metadata_path = f"{os.path.splitext(filepath)[0]}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            logger.info(f"ARIMAX model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving ARIMAX model: {e}")
            return False
    
    @classmethod
    def load_model(cls, filepath):
        """
        Load model from disk.
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            ARIMAXModel: Loaded model or None if error
        """
        try:
            # Load model using joblib
            model = joblib.load(filepath)
            logger.info(f"ARIMAX model loaded from {filepath}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading ARIMAX model: {e}")
            return None


class LightGBMModel:
    """LightGBM regression model for stock predictions."""
    
    def __init__(self, params=None):
        """
        Initialize LightGBM model with given parameters.
        
        Args:
            params (dict, optional): Model parameters. If None, default parameters are used.
        """
        # Default parameters
        self.params = params or {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'n_estimators': 100,
            'verbose': -1
        }
        
        self.model = None
        self.feature_names = None
        self.metadata = {
            'model_type': 'LightGBM',
            'parameters': self.params,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def fit(self, X, y, eval_set=None, early_stopping_rounds=None):
        """
        Fit LightGBM model.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            eval_set (list, optional): Validation set for early stopping
            early_stopping_rounds (int, optional): Early stopping rounds
            
        Returns:
            self: Fitted model
        """
        logger.info("Fitting LightGBM model")
        
        try:
            # Store feature names for prediction
            self.feature_names = X.columns.tolist()
            self.metadata['feature_names'] = self.feature_names
            
            # Create dataset
            train_data = lgb.Dataset(X, label=y)
            
            # Train model
            self.model = lgb.train(
                params=self.params,
                train_set=train_data,
                valid_sets=[train_data] if eval_set is None else [train_data, eval_set[0]],
                valid_names=['train'] if eval_set is None else ['train', 'valid'],
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=False
            )
            
            logger.info("LightGBM model training complete")
            
            # Store feature importances
            self.metadata['feature_importance'] = dict(zip(
                self.feature_names,
                self.model.feature_importance().tolist()
            ))
            
            return self
            
        except Exception as e:
            logger.error(f"Error fitting LightGBM model: {e}")
            raise
    
    def predict(self, X):
        """
        Make predictions with fitted LightGBM model.
        
        Args:
            X (pd.DataFrame): Feature matrix
            
        Returns:
            np.array: Predicted values
        """
        if self.model is None:
            logger.error("Model not fitted. Call fit() before predict()")
            return None
        
        try:
            # Ensure features match what was used for training
            if self.feature_names is not None:
                X = X[self.feature_names]
            
            # Make prediction
            return self.model.predict(X)
            
        except Exception as e:
            logger.error(f"Error making LightGBM prediction: {e}")
            return None
    
    def get_feature_importance(self, plot=False, top_n=20):
        """
        Get feature importance from the trained model.
        
        Args:
            plot (bool): Whether to plot feature importance
            top_n (int): Number of top features to show
            
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        if self.model is None:
            logger.error("Model not fitted. Call fit() before getting feature importance")
            return None
        
        try:
            # Get feature importance
            importance = self.model.feature_importance()
            feature_names = self.feature_names
            
            # Create dataframe
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            # Plot if requested
            if plot:
                plt.figure(figsize=(10, 6))
                plt.barh(importance_df['Feature'][:top_n], importance_df['Importance'][:top_n])
                plt.xlabel('Importance')
                plt.ylabel('Feature')
                plt.title(f'Top {top_n} Feature Importance')
                plt.tight_layout()
                plt.show()
            
            return importance_df
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return None
    
    def save_model(self, filepath):
        """
        Save fitted model to disk.
        
        Args:
            filepath (str): Path to save the model
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.model is None:
            logger.error("Model not fitted. Call fit() before saving")
            return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save LightGBM model
            self.model.save_model(filepath)
            
            # Save metadata and wrapper separately
            metadata_path = f"{os.path.splitext(filepath)[0]}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            wrapper_path = f"{os.path.splitext(filepath)[0]}_wrapper.pkl"
            joblib.dump(self, wrapper_path)
            
            logger.info(f"LightGBM model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving LightGBM model: {e}")
            return False
    
    @classmethod
    def load_model(cls, filepath):
        """
        Load model from disk.
        
        Args:
            filepath (str): Path to the saved wrapper
            
        Returns:
            LightGBMModel: Loaded model or None if error
        """
        try:
            # Load wrapper
            wrapper_path = f"{os.path.splitext(filepath)[0]}_wrapper.pkl"
            model = joblib.load(wrapper_path)
            
            # Load model if not included in the wrapper
            if not hasattr(model, 'model') or model.model is None:
                model.model = lgb.Booster(model_file=filepath)
                
            logger.info(f"LightGBM model loaded from {filepath}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading LightGBM model: {e}")
            return None


def train_arimax_model(features_df, target_col='return_1', exog_cols=None, 
                      order=(1, 0, 1), save_path=None):
    """
    Train an ARIMAX model on the given features.
    
    Args:
        features_df (pd.DataFrame): Features dataframe
        target_col (str): Target column name
        exog_cols (list): List of exogenous columns to use
        order (tuple): ARIMA order (p, d, q)
        save_path (str, optional): Path to save the model
        
    Returns:
        ARIMAXModel: Trained model
    """
    # Check input data
    if features_df.empty or target_col not in features_df.columns:
        logger.error("Invalid input data for ARIMAX training")
        return None
    
    # Get target variable
    y = features_df[target_col]
    
    # Get exogenous variables if specified
    X = None
    if exog_cols is not None and all(col in features_df.columns for col in exog_cols):
        X = features_df[exog_cols]
    
    # Create and fit model
    try:
        model = ARIMAXModel(order=order)
        model.fit(y, X)
        
        # Save model if path provided
        if save_path:
            model.save_model(save_path)
        
        return model
        
    except Exception as e:
        logger.error(f"Error training ARIMAX model: {e}")
        return None


def train_lightgbm_model(features_df, target_col='return_1', test_size=0.2,
                        params=None, save_path=None):
    """
    Train a LightGBM model on the given features.
    
    Args:
        features_df (pd.DataFrame): Features dataframe
        target_col (str): Target column name
        test_size (float): Test set size for validation
        params (dict, optional): LightGBM parameters
        save_path (str, optional): Path to save the model
        
    Returns:
        tuple: (LightGBMModel, metrics_dict)
    """
    # Prepare features
    X, y = prepare_model_features(features_df, target_col)
    
    if X.empty or y is None:
        logger.error("Invalid input data for LightGBM training")
        return None, None
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=config.RANDOM_STATE
    )
    
    # Create and fit model
    try:
        model = LightGBMModel(params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=50
        )
        
        # Evaluate model
        y_pred = model.predict(X_test)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        logger.info(f"LightGBM model metrics: {metrics}")
        
        # Save model if path provided
        if save_path:
            model.save_model(save_path)
        
        return model, metrics
        
    except Exception as e:
        logger.error(f"Error training LightGBM model: {e}")
        return None, None


def cross_validate_lightgbm(features_df, target_col='return_1', n_splits=5, params=None):
    """
    Perform time-series cross-validation for LightGBM model.
    
    Args:
        features_df (pd.DataFrame): Features dataframe
        target_col (str): Target column name
        n_splits (int): Number of cross-validation splits
        params (dict, optional): LightGBM parameters
        
    Returns:
        tuple: (mean_metrics_dict, list_of_metrics_per_fold)
    """
    # Prepare features
    X, y = prepare_model_features(features_df, target_col)
    
    if X.empty or y is None:
        logger.error("Invalid input data for LightGBM cross-validation")
        return None, None
    
    # Time series CV
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    metrics_list = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        logger.info(f"Training fold {fold+1}/{n_splits}")
        
        # Create and fit model
        try:
            model = LightGBMModel(params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                early_stopping_rounds=50
            )
            
            # Evaluate model
            y_pred = model.predict(X_test)
            
            fold_metrics = {
                'fold': fold + 1,
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
            
            metrics_list.append(fold_metrics)
            logger.info(f"Fold {fold+1} metrics: {fold_metrics}")
            
        except Exception as e:
            logger.error(f"Error in fold {fold+1}: {e}")
    
    # Calculate mean metrics
    if metrics_list:
        mean_metrics = {
            'mean_rmse': np.mean([m['rmse'] for m in metrics_list]),
            'mean_mae': np.mean([m['mae'] for m in metrics_list]),
            'mean_r2': np.mean([m['r2'] for m in metrics_list]),
            'std_rmse': np.std([m['rmse'] for m in metrics_list]),
            'std_mae': np.std([m['mae'] for m in metrics_list]),
            'std_r2': np.std([m['r2'] for m in metrics_list])
        }
        logger.info(f"Cross-validation mean metrics: {mean_metrics}")
        return mean_metrics, metrics_list
    
    return None, None


def train_model_pipeline(price_df, sentiment_df=None, model_type='lightgbm',
                        target_col='return_1', save_dir='models'):
    """
    End-to-end pipeline to prepare data and train a model.
    
    Args:
        price_df (pd.DataFrame): Price dataframe with OHLCV data
        sentiment_df (pd.DataFrame, optional): Sentiment dataframe
        model_type (str): Type of model to train ('lightgbm' or 'arimax')
        target_col (str): Target column to predict
        save_dir (str): Directory to save the model
        
    Returns:
        tuple: (trained_model, metrics)
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate timestamp for model versioning
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Prepare features
    logger.info("Preparing features...")
    from features.feature_aggregator import FeatureAggregator
    
    aggregator = FeatureAggregator()
    features_df = aggregator.prepare_complete_features(price_df, sentiment_df)
    
    if features_df.empty:
        logger.error("Failed to generate features")
        return None, None
    
    # Train model based on type
    if model_type.lower() == 'lightgbm':
        logger.info("Training LightGBM model...")
        save_path = os.path.join(save_dir, f"lightgbm_{timestamp}.txt")
        return train_lightgbm_model(features_df, target_col=target_col, save_path=save_path)
    
    elif model_type.lower() == 'arimax':
        logger.info("Training ARIMAX model...")
        save_path = os.path.join(save_dir, f"arimax_{timestamp}.pkl")
        
        # Select a subset of the most important features for ARIMAX
        # (ARIMAX doesn't handle high-dimensional data well)
        rsi_cols = [col for col in features_df.columns if col.startswith('rsi_')]
        sentiment_cols = [col for col in features_df.columns if 'compound_mean' in col]
        exog_cols = rsi_cols + sentiment_cols
        
        model = train_arimax_model(
            features_df, 
            target_col=target_col, 
            exog_cols=exog_cols, 
            save_path=save_path
        )
        return model, None
    
    else:
        logger.error(f"Unsupported model type: {model_type}")
        return None, None


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Create sample price data
    dates = pd.date_range(start='2023-01-01', periods=500, freq='1h')
    np.random.seed(42)
    
    # Random walk price series with some seasonality
    closes = [100]
    for i in range(1, 500):
        # Add some seasonality
        seasonal = 0.5 * np.sin(i * 2 * np.pi / 24)  # 24-hour seasonality
        # Add trend and noise
        closes.append(closes[-1] * (1 + np.random.normal(0, 0.005) + 0.0005 + seasonal * 0.01))
    
    price_df = pd.DataFrame({
        'timestamp': dates,
        'symbol': 'EXAMPLE',
        'open': closes,
        'high': [c * (1 + abs(np.random.normal(0, 0.002))) for c in closes],
        'low': [c * (1 - abs(np.random.normal(0, 0.002))) for c in closes],
        'close': closes,
        'volume': np.random.randint(1000, 10000, 500)
    })
    
    # Prepare features
    from features.technical_indicators import prepare_features
    features_df = prepare_features(price_df)
    
    print(f"Prepared feature set with shape: {features_df.shape}")
    
    # Train ARIMAX model
    print("\nTraining ARIMAX model...")
    exog_cols = ['rsi_14', 'macd_line', 'bb_20_width']
    arimax_model = train_arimax_model(
        features_df, 
        target_col='return_1',
        exog_cols=exog_cols,
        order=(2, 0, 2)
    )
    if arimax_model is not None:
        print("ARIMAX model training successful")
    
    # Train LightGBM model
    print("\nTraining LightGBM model...")
    lgb_model, metrics = train_lightgbm_model(features_df, target_col='return_1')
    if lgb_model is not None:
        print(f"LightGBM training metrics: {metrics}")
        
        # Show feature importance
        importance_df = lgb_model.get_feature_importance(plot=False, top_n=10)
        print("\nTop 10 feature importance:")
        print(importance_df.head(10))
    
    # Cross-validation
    print("\nPerforming cross-validation...")
    cv_metrics, fold_metrics = cross_validate_lightgbm(features_df, target_col='return_1', n_splits=3)
    if cv_metrics is not None:
        print(f"Cross-validation metrics: {cv_metrics}")
    
    # Save and load model test
    if lgb_model is not None:
        print("\nTesting model save/load functionality...")
        os.makedirs("models", exist_ok=True)
        model_path = "models/test_model.txt"
        
        # Save
        lgb_model.save_model(model_path)
        
        # Load
        loaded_model = LightGBMModel.load_model(model_path)
        if loaded_model is not None:
            print("Model successfully loaded") 