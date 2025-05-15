"""
Client for fetching delayed stock price data from Polygon.io free tier.

The free tier of Polygon.io provides delayed market data (15 min) with
intraday data at 1-minute intervals.
"""
import os
import json
import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timedelta
import sys

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.logger import setup_logger
import config

logger = setup_logger('polygon_client')

class PolygonClient:
    """Client for Polygon.io API to fetch stock price data."""
    
    def __init__(self, api_key=None):
        """
        Initialize the Polygon API client.
        
        Args:
            api_key (str, optional): Polygon API key. If not provided,
                                   it will be read from config.
        """
        self.api_key = api_key or config.POLYGON_API_KEY
        self.base_url = "https://api.polygon.io"
        if not self.api_key:
            logger.error("Polygon API key not provided or found in config")
            raise ValueError("Polygon API key is required")
    
    async def get_aggregates(self, symbol, multiplier=1, timespan='minute', 
                           from_date=None, to_date=None, limit=120):
        """
        Fetch aggregate bars for a stock over a given date range.
        
        Args:
            symbol (str): Stock ticker symbol
            multiplier (int): Size of the timespan multiplier (default: 1)
            timespan (str): Size of the time window (minute, hour, day, etc.)
            from_date (str): Start date in the format YYYY-MM-DD
            to_date (str): End date in the format YYYY-MM-DD
            limit (int): Number of results to return (max 50000)
            
        Returns:
            pd.DataFrame: DataFrame with OHLCV data
        """
        if not from_date:
            # Default to yesterday if not specified
            from_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        if not to_date:
            # Default to today if not specified
            to_date = datetime.now().strftime('%Y-%m-%d')
        
        url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        params = {
            "apiKey": self.api_key,
            "limit": limit,
            "adjusted": "true"
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._process_aggregates(data, symbol)
                    else:
                        error_text = await response.text()
                        logger.error(f"Error fetching data for {symbol}: {response.status} - {error_text}")
                        return None
            except Exception as e:
                logger.error(f"Exception while fetching data for {symbol}: {e}")
                return None
    
    def _process_aggregates(self, response_data, symbol):
        """
        Process the raw API response into a pandas DataFrame.
        
        Args:
            response_data (dict): JSON response from Polygon API
            symbol (str): Stock ticker symbol
            
        Returns:
            pd.DataFrame: DataFrame with processed OHLCV data
        """
        if not response_data.get('results'):
            logger.warning(f"No aggregate data found for {symbol}")
            return pd.DataFrame()
        
        df = pd.DataFrame(response_data['results'])
        
        # Rename columns to standard OHLCV format
        df = df.rename(columns={
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume',
            't': 'timestamp',
            'n': 'transactions'
        })
        
        # Convert timestamp from milliseconds to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Add symbol column
        df['symbol'] = symbol
        
        # Select and reorder columns
        cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        return df[cols].sort_values('timestamp')

    async def get_last_quote(self, symbol):
        """
        Fetch the last quote for a symbol.
        
        Args:
            symbol (str): Stock ticker symbol
            
        Returns:
            dict: Quote data or None if not available
        """
        url = f"{self.base_url}/v2/last/nbbo/{symbol}"
        params = {"apiKey": self.api_key}
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('status') == 'OK' and data.get('results'):
                            return data['results']
                        else:
                            logger.warning(f"No quote data found for {symbol}")
                            return None
                    else:
                        logger.error(f"Error fetching quote for {symbol}: {response.status}")
                        return None
            except Exception as e:
                logger.error(f"Exception while fetching quote for {symbol}: {e}")
                return None

async def get_multiple_symbols(symbols, days_back=1):
    """
    Fetch data for multiple symbols in parallel.
    
    Args:
        symbols (list): List of stock ticker symbols
        days_back (int): Number of days to look back
        
    Returns:
        dict: Dictionary mapping each symbol to its DataFrame
    """
    client = PolygonClient()
    from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    to_date = datetime.now().strftime('%Y-%m-%d')
    
    tasks = []
    for symbol in symbols:
        task = client.get_aggregates(
            symbol, 
            from_date=from_date, 
            to_date=to_date
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return {symbol: df for symbol, df in zip(symbols, results) if df is not None and not df.empty}

if __name__ == "__main__":
    # Example usage
    async def main():
        # Get data for a single symbol
        client = PolygonClient()
        df = await client.get_aggregates("AAPL", limit=10)
        if df is not None and not df.empty:
            print(f"AAPL data:\n{df.head()}")
        
        # Get data for multiple symbols
        symbols = ["MSFT", "GOOGL", "AMZN"]
        results = await get_multiple_symbols(symbols)
        for symbol, data in results.items():
            print(f"\n{symbol} data:\n{data.head() if not data.empty else 'No data'}")

    asyncio.run(main()) 