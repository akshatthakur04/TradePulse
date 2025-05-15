"""
Configuration for the TradePulse stock prediction pipeline.
Contains API keys, URLs, and parameter settings.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys (set these in your .env file or environment variables)
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "")
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "")
STOCKTWITS_API_KEY = os.environ.get("STOCKTWITS_API_KEY", "")
REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.environ.get("REDDIT_USER_AGENT", "TradePulse:v1.0 (by /u/your_username)")

# Redis Configuration
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/0"

# Stream Names
PRICE_STREAM = "tradepulse:prices"
NEWS_STREAM = "tradepulse:news"
SOCIAL_STREAM = "tradepulse:social"
SENTIMENT_STREAM = "tradepulse:sentiment"
FEATURE_STREAM = "tradepulse:features"
PREDICTION_STREAM = "tradepulse:predictions"

# Data Polling Intervals (in seconds)
PRICE_POLL_INTERVAL = 60  # 1 minute
NEWS_POLL_INTERVAL = 300  # 5 minutes
SOCIAL_POLL_INTERVAL = 300  # 5 minutes

# Model Configuration
DEFAULT_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "AMD", "JPM", "BAC"]
LOOKBACK_WINDOW = 30  # 30 minutes for feature calculations
PREDICTION_HORIZON = 1  # Predict 1 minute ahead

# Sentiment Analysis
VADER_THRESHOLD_POSITIVE = 0.05
VADER_THRESHOLD_NEGATIVE = -0.05

# Technical Indicator Parameters
SMA_PERIODS = [5, 10, 20]
EMA_PERIODS = [5, 10, 20]
RSI_PERIOD = 14
BOLLINGER_PERIOD = 20
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Model Training
TRAIN_TEST_SPLIT = 0.8
CV_FOLDS = 5
RANDOM_STATE = 42 