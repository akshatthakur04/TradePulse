"""
Price worker that processes market data and computes technical indicators.

This worker polls price data from APIs, computes technical indicators,
and publishes the results to Redis Streams for feature generation.
"""
import os
import sys
import asyncio
import pandas as pd
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.logger import setup_logger
import config
from data_ingestion.polygon_client import get_multiple_symbols
from data_ingestion.alphavantage_client import get_multiple_quotes
from features.technical_indicators import TechnicalIndicators
from orchestrator.redis_streams import RedisStreamClient, consume_stream

logger = setup_logger('worker_price')

class PriceWorker:
    """Worker that processes price data and computes technical indicators."""
    
    def __init__(self, redis_client=None, symbols=None):
        """
        Initialize the price worker.
        
        Args:
            redis_client (RedisStreamClient, optional): Redis client instance
            symbols (list, optional): List of symbols to process
        """
        self.redis_client = redis_client or RedisStreamClient()
        self.symbols = symbols or config.DEFAULT_SYMBOLS
        self.price_stream = config.PRICE_STREAM
        self.feature_stream = config.FEATURE_STREAM
        
        # Initialize streams
        self.redis_client.create_stream(self.price_stream)
        self.redis_client.create_stream(self.feature_stream)
        
        # Technical indicators calculator
        self.tech_indicators = TechnicalIndicators()
        
        # Cache for price data
        self.price_cache = {symbol: pd.DataFrame() for symbol in self.symbols}
        self.cache_lookback = timedelta(minutes=config.LOOKBACK_WINDOW * 2)  # 2x lookback for safety
        
        logger.info(f"Price worker initialized for symbols: {self.symbols}")
    
    async def fetch_prices(self, use_fallback=False):
        """
        Fetch price data for all symbols.
        
        Args:
            use_fallback (bool): Whether to use Alpha Vantage as fallback
            
        Returns:
            dict: Dictionary mapping symbols to price DataFrames
        """
        try:
            logger.info(f"Fetching price data for {len(self.symbols)} symbols")
            
            # Primary source: Polygon.io
            if not use_fallback:
                price_dict = await get_multiple_symbols(self.symbols, days_back=1)
                
                # Check if we got data for all symbols
                missing_symbols = [s for s in self.symbols if s not in price_dict or price_dict[s].empty]
                
                if not missing_symbols:
                    logger.info(f"Successfully fetched price data from Polygon.io")
                    return price_dict
                else:
                    logger.warning(f"Missing price data for {missing_symbols}, using fallback")
                    use_fallback = True
            
            # Fallback source: Alpha Vantage
            if use_fallback:
                # Get quotes from Alpha Vantage
                quotes = await get_multiple_quotes(self.symbols)
                
                # If no quotes, return empty dict
                if not quotes:
                    logger.error("Failed to get quotes from fallback source")
                    return {}
                
                # Convert quotes to price dataframes
                price_dict = {}
                for symbol, quote in quotes.items():
                    if not quote:
                        continue
                    
                    # Create a single-row dataframe with the quote data
                    df = pd.DataFrame([{
                        'timestamp': datetime.now(),
                        'symbol': symbol,
                        'open': float(quote.get('price', 0)),
                        'high': float(quote.get('price', 0)),
                        'low': float(quote.get('price', 0)),
                        'close': float(quote.get('price', 0)),
                        'volume': int(quote.get('volume', 0))
                    }])
                    
                    price_dict[symbol] = df
                
                logger.info(f"Fetched {len(price_dict)} quotes from Alpha Vantage")
            
            return price_dict
            
        except Exception as e:
            logger.error(f"Error fetching prices: {e}")
            return {}
    
    def update_price_cache(self, new_prices):
        """
        Update the price cache with new data.
        
        Args:
            new_prices (dict): Dictionary mapping symbols to price DataFrames
        """
        for symbol, df in new_prices.items():
            if symbol not in self.price_cache:
                self.price_cache[symbol] = df
            else:
                # Concatenate and drop duplicates
                combined = pd.concat([self.price_cache[symbol], df])
                combined = combined.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
                
                # Trim old data
                cutoff = datetime.now() - self.cache_lookback
                combined = combined[combined['timestamp'] > cutoff]
                
                self.price_cache[symbol] = combined
    
    def compute_indicators(self, price_df):
        """
        Compute technical indicators for a price DataFrame.
        
        Args:
            price_df (pd.DataFrame): DataFrame with OHLCV price data
            
        Returns:
            pd.DataFrame: DataFrame with added indicators
        """
        try:
            # Add all indicators
            with_indicators = self.tech_indicators.add_all_indicators(price_df)
            return with_indicators
            
        except Exception as e:
            logger.error(f"Error computing indicators: {e}")
            return price_df
    
    async def publish_prices(self, price_dict):
        """
        Publish price data to the price stream.
        
        Args:
            price_dict (dict): Dictionary mapping symbols to price DataFrames
            
        Returns:
            int: Number of published price messages
        """
        count = 0
        
        for symbol, df in price_dict.items():
            if df.empty:
                continue
                
            # Only publish the most recent price point
            latest = df.iloc[-1]
            
            # Add to price stream
            msg_id = self.redis_client.add_dataframe_row(self.price_stream, latest)
            if msg_id:
                count += 1
                logger.debug(f"Published price for {symbol}: {latest['close']}")
        
        if count > 0:
            logger.info(f"Published {count} price updates to stream")
            
        return count
    
    async def publish_features(self, feature_dict):
        """
        Publish feature data to the feature stream.
        
        Args:
            feature_dict (dict): Dictionary mapping symbols to feature DataFrames
            
        Returns:
            int: Number of published feature messages
        """
        count = 0
        
        for symbol, df in feature_dict.items():
            if df.empty:
                continue
                
            # Only publish the most recent feature point
            latest = df.iloc[-1]
            
            # Add to feature stream
            msg_id = self.redis_client.add_dataframe_row(self.feature_stream, latest)
            if msg_id:
                count += 1
                logger.debug(f"Published features for {symbol}")
        
        if count > 0:
            logger.info(f"Published {count} feature sets to stream")
            
        return count
    
    async def run_once(self):
        """
        Execute a single run of the price processing pipeline.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Fetch prices
            price_dict = await self.fetch_prices()
            
            if not price_dict:
                logger.warning("No price data fetched")
                return False
                
            # Update cache
            self.update_price_cache(price_dict)
            
            # Publish raw prices
            await self.publish_prices(price_dict)
            
            # Compute features for each symbol
            feature_dict = {}
            for symbol, df in self.price_cache.items():
                if not df.empty and len(df) > 10:  # Need enough data for indicators
                    features = self.compute_indicators(df)
                    feature_dict[symbol] = features
            
            # Publish features
            await self.publish_features(feature_dict)
            
            return True
            
        except Exception as e:
            logger.error(f"Error in price worker run: {e}")
            return False
    
    async def run_continuously(self, interval_seconds=60):
        """
        Run the price worker continuously at specified intervals.
        
        Args:
            interval_seconds (int): Interval between runs in seconds
        """
        logger.info(f"Starting price worker with {interval_seconds}s interval")
        
        while True:
            try:
                await self.run_once()
                logger.info(f"Waiting {interval_seconds}s until next run")
                await asyncio.sleep(interval_seconds)
                
            except asyncio.CancelledError:
                logger.info("Price worker cancelled")
                break
                
            except Exception as e:
                logger.error(f"Error in price worker loop: {e}")
                await asyncio.sleep(10)  # Sleep a short time on error

async def process_price_message(message):
    """
    Process a single price message from the stream.
    
    Args:
        message (dict): Price message to process
    """
    try:
        # Extract price data
        price = {}
        for key, value in message.items():
            if key != '_id':  # Skip message ID
                price[key] = value
        
        # Required fields
        required = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(field in price for field in required):
            logger.error(f"Missing required fields in price message: {price}")
            return
            
        symbol = price['symbol']
        logger.debug(f"Processing price message for {symbol}")
        
        # Convert types
        price['timestamp'] = pd.to_datetime(price['timestamp'])
        for field in ['open', 'high', 'low', 'close']:
            price[field] = float(price[field])
        price['volume'] = int(price['volume'])
        
        # Create a DataFrame with a single row
        price_df = pd.DataFrame([price])
        
        # Create worker and compute indicators
        redis_client = RedisStreamClient()
        worker = PriceWorker(redis_client, symbols=[symbol])
        
        # Add to cache
        worker.update_price_cache({symbol: price_df})
        
        # Compute indicators for the entire cached history
        if symbol in worker.price_cache and not worker.price_cache[symbol].empty:
            features = worker.compute_indicators(worker.price_cache[symbol])
            
            # Publish to feature stream
            if not features.empty:
                await worker.publish_features({symbol: features})
        
    except Exception as e:
        logger.error(f"Error processing price message: {e}")

async def start_price_worker_service():
    """Start the price worker service."""
    # Create Redis client and worker
    redis_client = RedisStreamClient()
    worker = PriceWorker(redis_client)
    
    # Create tasks
    polling_task = asyncio.create_task(
        worker.run_continuously(interval_seconds=config.PRICE_POLL_INTERVAL)
    )
    
    consumer_task = asyncio.create_task(
        consume_stream(
            redis_client,
            config.PRICE_STREAM,
            "price_processors",
            "indicator_calculator",
            process_func=process_price_message
        )
    )
    
    # Run until cancelled
    try:
        await asyncio.gather(polling_task, consumer_task)
    except asyncio.CancelledError:
        polling_task.cancel()
        consumer_task.cancel()
        await asyncio.gather(polling_task, consumer_task, return_exceptions=True)
        logger.info("Price worker service stopped")

if __name__ == "__main__":
    # Example usage
    async def main():
        # Create worker
        redis_client = RedisStreamClient()
        worker = PriceWorker(redis_client, symbols=['AAPL', 'MSFT', 'GOOGL'])
        
        # Run once
        print("Running price worker once...")
        success = await worker.run_once()
        print(f"Run completed with success: {success}")
        
        # If successful, print some info
        if success:
            for symbol, df in worker.price_cache.items():
                print(f"\n{symbol} price cache: {len(df)} entries")
                if not df.empty:
                    print(df[['timestamp', 'close']].tail(3))
        
        # Test continuous mode for 30 seconds with short interval
        print("\nRunning price worker continuously for 30 seconds...")
        continuous_task = asyncio.create_task(worker.run_continuously(interval_seconds=10))
        
        # Sleep for 30 seconds
        await asyncio.sleep(30)
        
        # Cancel task
        continuous_task.cancel()
        try:
            await continuous_task
        except asyncio.CancelledError:
            print("Continuous task cancelled")
        
        print("Example completed")
    
    asyncio.run(main()) 