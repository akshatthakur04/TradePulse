"""
Client for fetching stock price data from Alpha Vantage API as a fallback data source.

Alpha Vantage provides a free tier API for stock data with some rate limiting.
"""
import os
import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timedelta
import sys

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.logger import setup_logger
from config.api_config import get_api_config, validate_api_key

logger = setup_logger('alphavantage_client')

class AlphaVantageClient:
    """Client for Alpha Vantage API to fetch stock price data."""
    
    def __init__(self, api_key=None):
        """
        Initialize the Alpha Vantage API client.
        
        Args:
            api_key (str, optional): Alpha Vantage API key. If not provided,
                                     it will be read from config.
        """
        api_config = get_api_config('ALPHA_VANTAGE')
        if not api_config:
            logger.error("Alpha Vantage API configuration not found")
            raise ValueError("Alpha Vantage API configuration not found")
            
        self.api_key = api_key or api_config.get('api_key')
        self.base_url = api_config.get('base_url')
        
        if not self.api_key:
            logger.error("Alpha Vantage API key not provided or found in config")
            raise ValueError("Alpha Vantage API key is required")
        
        # Get rate limit settings from config
        rate_limit = api_config.get('rate_limit', {})
        self.request_interval = rate_limit.get('interval_seconds', 12)
        self.last_request_time = datetime.now() - timedelta(seconds=self.request_interval)
    
    async def _rate_limited_request(self, url, params):
        """
        Make a rate-limited request to Alpha Vantage API.
        
        Args:
            url (str): The endpoint URL
            params (dict): Request parameters
            
        Returns:
            dict: JSON response or None on error
        """
        # Check if we need to wait to respect rate limits
        time_since_last = (datetime.now() - self.last_request_time).total_seconds()
        if time_since_last < self.request_interval:
            wait_time = self.request_interval - time_since_last
            logger.debug(f"Rate limiting: Waiting {wait_time:.2f} seconds")
            await asyncio.sleep(wait_time)
        
        self.last_request_time = datetime.now()
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        logger.error(f"Error with Alpha Vantage API: {response.status} - {error_text}")
                        return None
            except Exception as e:
                logger.error(f"Exception while calling Alpha Vantage API: {e}")
                return None
    
    async def get_intraday(self, symbol, interval='1min', outputsize='compact'):
        """
        Fetch intraday time series data for a given symbol.
        
        Args:
            symbol (str): Stock ticker symbol
            interval (str): Time interval between data points (1min, 5min, 15min, 30min, 60min)
            outputsize (str): 'compact' returns latest 100 points, 'full' returns up to 20 years of data
            
        Returns:
            pd.DataFrame: DataFrame with OHLCV data or empty DataFrame if error
        """
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": self.api_key
        }
        
        data = await self._rate_limited_request(self.base_url, params)
        if not data:
            return pd.DataFrame()
        
        # Check for error messages
        if "Error Message" in data:
            logger.error(f"Alpha Vantage error for {symbol}: {data['Error Message']}")
            return pd.DataFrame()
        
        # Check if we have the time series data
        time_series_key = f"Time Series ({interval})"
        if time_series_key not in data:
            logger.error(f"Time series data not found for {symbol}")
            return pd.DataFrame()
        
        # Convert the nested dict to DataFrame
        time_series = data[time_series_key]
        df = pd.DataFrame.from_dict(time_series, orient="index")
        
        # Rename columns
        df = df.rename(columns={
            '1. open': 'open',
            '2. high': 'high',
            '3. low': 'low',
            '4. close': 'close',
            '5. volume': 'volume'
        })
        
        # Convert types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        
        # Reset index to convert timestamp to column
        df = df.reset_index().rename(columns={'index': 'timestamp'})
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Add symbol column
        df['symbol'] = symbol
        
        # Reorder columns
        cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        return df[cols].sort_values('timestamp')
    
    async def get_quote(self, symbol):
        """
        Fetch the latest global quote for a symbol.
        
        Args:
            symbol (str): Stock ticker symbol
            
        Returns:
            dict: Dictionary with quote data or None if error
        """
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": self.api_key
        }
        
        data = await self._rate_limited_request(self.base_url, params)
        if not data or "Global Quote" not in data or not data["Global Quote"]:
            return None
        
        quote = data["Global Quote"]
        return {
            'symbol': quote.get('01. symbol'),
            'price': float(quote.get('05. price', 0)),
            'change': float(quote.get('09. change', 0)),
            'change_percent': quote.get('10. change percent', '0%'),
            'volume': int(quote.get('06. volume', 0)),
            'latest_trading_day': quote.get('07. latest trading day')
        }

async def get_multiple_quotes(symbols):
    """
    Fetch quotes for multiple symbols, respecting rate limits.
    
    Args:
        symbols (list): List of stock ticker symbols
        
    Returns:
        dict: Dictionary mapping symbols to their quote data
    """
    client = AlphaVantageClient()
    results = {}
    
    for symbol in symbols:
        quote = await client.get_quote(symbol)
        if quote:
            results[symbol] = quote
    
    return results

if __name__ == "__main__":
    # Example usage
    async def main():
        # Check if API key is available
        if not validate_api_key('ALPHA_VANTAGE'):
            logger.error("Alpha Vantage API key not set. Please set the ALPHA_VANTAGE_API_KEY environment variable.")
            return
            
        # Get intraday data for a symbol
        client = AlphaVantageClient()
        df = await client.get_intraday("AAPL")
        if not df.empty:
            print(f"AAPL intraday data:\n{df.head()}")
        
        # Get quotes for multiple symbols
        symbols = ["MSFT", "GOOGL"]
        quotes = await get_multiple_quotes(symbols)
        for symbol, quote in quotes.items():
            print(f"\n{symbol} quote: {quote}")
    
    asyncio.run(main()) 