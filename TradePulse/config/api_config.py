"""
API Configuration File

This module provides centralized access to all external API configurations.
API keys and sensitive data should be loaded from environment variables or a secure vault.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Alpha Vantage API Configuration
ALPHA_VANTAGE = {
    "api_key": os.environ.get("ALPHA_VANTAGE_API_KEY", ""),
    "base_url": "https://www.alphavantage.co/query",
    "rate_limit": {
        "requests_per_minute": 5,
        "requests_per_day": 500,
        "interval_seconds": 12
    }
}

# Yahoo Finance API Configuration (if used)
YAHOO_FINANCE = {
    "base_url": "https://query1.finance.yahoo.com/v8/finance",
    "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}

# Polygon.io API Configuration (if used)
POLYGON = {
    "api_key": os.environ.get("POLYGON_API_KEY", ""),
    "base_url": "https://api.polygon.io"
}

# Add other API configurations as needed

def get_api_config(api_name):
    """
    Get the configuration for a specific API.
    
    Args:
        api_name (str): Name of the API (e.g., 'ALPHA_VANTAGE', 'YAHOO_FINANCE')
    
    Returns:
        dict: API configuration dictionary or None if not found
    """
    return globals().get(api_name.upper())

def validate_api_key(api_name):
    """
    Validate that an API key exists for the specified API.
    
    Args:
        api_name (str): Name of the API
    
    Returns:
        bool: True if the API key exists and is not empty, False otherwise
    """
    config = get_api_config(api_name)
    if not config or not config.get("api_key"):
        return False
    return True 