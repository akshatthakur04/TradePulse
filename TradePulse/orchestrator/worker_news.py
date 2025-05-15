"""
News worker that processes news feeds and produces sentiment.

This worker polls news sources, extracts sentiment using VADER and FinBERT,
and publishes the results to Redis Streams for feature generation.
"""
import os
import sys
import asyncio
import pandas as pd
import json
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.logger import setup_logger
import config
from data_ingestion.rss_feed import get_recent_financial_news
from sentiment.vader_sentiment import process_news_sentiment as process_vader_sentiment
from sentiment.finbert_sentiment import process_news_sentiment as process_finbert_sentiment
from orchestrator.redis_streams import RedisStreamClient, consume_stream

logger = setup_logger('worker_news')

class NewsWorker:
    """Worker that processes news feeds and extracts sentiment."""
    
    def __init__(self, redis_client=None):
        """
        Initialize the news worker.
        
        Args:
            redis_client (RedisStreamClient, optional): Redis client instance
        """
        self.redis_client = redis_client or RedisStreamClient()
        self.news_stream = config.NEWS_STREAM
        self.sentiment_stream = config.SENTIMENT_STREAM
        
        # Initialize streams
        self.redis_client.create_stream(self.news_stream)
        self.redis_client.create_stream(self.sentiment_stream)
        
        # Last processed timestamp to avoid duplicates
        self.last_processed_time = datetime.now() - timedelta(hours=24)
        
        logger.info("News worker initialized")
    
    async def fetch_news(self):
        """
        Fetch news from RSS feeds and publish to news stream.
        
        Returns:
            pd.DataFrame: DataFrame of fetched news or None if error
        """
        try:
            logger.info("Fetching news from feeds")
            news_df = await get_recent_financial_news(max_articles=10)
            
            if news_df.empty:
                logger.warning("No news found")
                return None
            
            # Convert published dates to datetime if needed
            if 'published' in news_df.columns:
                news_df['published'] = pd.to_datetime(news_df['published'])
            
            # Filter out already processed news
            if 'published' in news_df.columns:
                news_df = news_df[news_df['published'] > self.last_processed_time]
            
            if news_df.empty:
                logger.info("No new news to process")
                return None
            
            # Update last processed time
            if 'published' in news_df.columns and not news_df.empty:
                self.last_processed_time = news_df['published'].max()
            
            # Publish to news stream
            news_count = 0
            for _, row in news_df.iterrows():
                # Add timestamp if missing
                if 'timestamp' not in row:
                    row['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Add row to stream
                msg_id = self.redis_client.add_dataframe_row(self.news_stream, row)
                if msg_id:
                    news_count += 1
            
            logger.info(f"Published {news_count} news items to stream")
            return news_df
            
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return None
    
    async def process_news_sentiment(self, news_df):
        """
        Process news sentiment and publish to sentiment stream.
        
        Args:
            news_df (pd.DataFrame): DataFrame of news items
            
        Returns:
            pd.DataFrame: DataFrame with sentiment scores
        """
        if news_df is None or news_df.empty:
            return None
        
        try:
            logger.info("Processing news sentiment with VADER")
            # Process with VADER
            news_with_vader = await process_vader_sentiment(news_df, text_column='title')
            
            # If full_text is available, use FinBERT for deeper analysis
            if 'full_text' in news_df.columns:
                logger.info("Processing news sentiment with FinBERT")
                full_text_items = news_df[news_df['full_text'].notna() & (news_df['full_text'] != '')]
                
                if not full_text_items.empty:
                    news_with_finbert = await process_finbert_sentiment(full_text_items, text_column='full_text')
                    
                    # Merge FinBERT results back
                    if news_with_finbert is not None and not news_with_finbert.empty:
                        # Extract only the FinBERT columns
                        finbert_cols = [col for col in news_with_finbert.columns if col.startswith('finbert_')]
                        finbert_df = news_with_finbert[['title'] + finbert_cols]
                        
                        # Merge with VADER results
                        news_with_sentiment = pd.merge(news_with_vader, finbert_df, on='title', how='left')
                    else:
                        news_with_sentiment = news_with_vader
                else:
                    news_with_sentiment = news_with_vader
            else:
                news_with_sentiment = news_with_vader
            
            # Extract symbols from article if available
            if 'symbols' in news_with_sentiment.columns:
                # For each news item with symbols
                for _, row in news_with_sentiment.iterrows():
                    if not row['symbols']:
                        continue
                    
                    # If symbols is a string, parse it
                    symbols = row['symbols']
                    if isinstance(symbols, str):
                        try:
                            symbols = json.loads(symbols)
                        except:
                            # If it's a comma-separated string
                            symbols = [s.strip() for s in symbols.split(',')]
                    
                    # For each symbol, create a sentiment entry
                    for symbol in symbols:
                        sentiment_entry = {
                            'symbol': symbol,
                            'timestamp': row.get('published', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                            'source': row.get('source', 'news'),
                            'title': row.get('title', ''),
                            'url': row.get('link', ''),
                            'vader_compound': row.get('vader_compound', 0),
                            'vader_sentiment': row.get('vader_sentiment', 'neutral')
                        }
                        
                        # Add FinBERT sentiment if available
                        if 'finbert_sentiment' in row:
                            sentiment_entry['finbert_sentiment'] = row.get('finbert_sentiment')
                            sentiment_entry['finbert_compound'] = row.get('finbert_compound', 0)
                        
                        # Add to sentiment stream
                        self.redis_client.add_message(self.sentiment_stream, sentiment_entry)
            
            # Always publish general market sentiment (not symbol-specific)
            for _, row in news_with_sentiment.iterrows():
                sentiment_entry = {
                    'symbol': 'MARKET',
                    'timestamp': row.get('published', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                    'source': row.get('source', 'news'),
                    'title': row.get('title', ''),
                    'url': row.get('link', ''),
                    'vader_compound': row.get('vader_compound', 0),
                    'vader_sentiment': row.get('vader_sentiment', 'neutral')
                }
                
                # Add FinBERT sentiment if available
                if 'finbert_sentiment' in row:
                    sentiment_entry['finbert_sentiment'] = row.get('finbert_sentiment')
                    sentiment_entry['finbert_compound'] = row.get('finbert_compound', 0)
                
                # Add to sentiment stream
                self.redis_client.add_message(self.sentiment_stream, sentiment_entry)
            
            logger.info(f"Published sentiment for {len(news_with_sentiment)} news items")
            return news_with_sentiment
            
        except Exception as e:
            logger.error(f"Error processing news sentiment: {e}")
            return None
    
    async def run_once(self):
        """
        Execute a single run of the news processing pipeline.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Fetch news
            news_df = await self.fetch_news()
            
            # Process sentiment
            if news_df is not None and not news_df.empty:
                await self.process_news_sentiment(news_df)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error in news worker run: {e}")
            return False
    
    async def run_continuously(self, interval_seconds=300):
        """
        Run the news worker continuously at specified intervals.
        
        Args:
            interval_seconds (int): Interval between runs in seconds
        """
        logger.info(f"Starting news worker with {interval_seconds}s interval")
        
        while True:
            try:
                await self.run_once()
                logger.info(f"Waiting {interval_seconds}s until next run")
                await asyncio.sleep(interval_seconds)
                
            except asyncio.CancelledError:
                logger.info("News worker cancelled")
                break
                
            except Exception as e:
                logger.error(f"Error in news worker loop: {e}")
                await asyncio.sleep(10)  # Sleep a short time on error

async def process_news_message(message):
    """
    Process a single news message from the stream.
    
    Args:
        message (dict): News message to process
    """
    try:
        # Extract news data
        news = {}
        for key, value in message.items():
            if key != '_id':  # Skip message ID
                news[key] = value
        
        logger.debug(f"Processing news: {news.get('title', '')}")
        
        # Create a DataFrame with a single row
        news_df = pd.DataFrame([news])
        
        # Process sentiment
        redis_client = RedisStreamClient()
        worker = NewsWorker(redis_client)
        
        # Process and publish sentiment
        await worker.process_news_sentiment(news_df)
        
    except Exception as e:
        logger.error(f"Error processing news message: {e}")

async def start_news_worker_service():
    """Start the news worker service."""
    # Create Redis client and worker
    redis_client = RedisStreamClient()
    worker = NewsWorker(redis_client)
    
    # Create tasks
    polling_task = asyncio.create_task(
        worker.run_continuously(interval_seconds=config.NEWS_POLL_INTERVAL)
    )
    
    consumer_task = asyncio.create_task(
        consume_stream(
            redis_client,
            config.NEWS_STREAM,
            "news_processors",
            "sentiment_extractor",
            process_func=process_news_message
        )
    )
    
    # Run until cancelled
    try:
        await asyncio.gather(polling_task, consumer_task)
    except asyncio.CancelledError:
        polling_task.cancel()
        consumer_task.cancel()
        await asyncio.gather(polling_task, consumer_task, return_exceptions=True)
        logger.info("News worker service stopped")

if __name__ == "__main__":
    # Example usage
    async def main():
        # Create worker
        redis_client = RedisStreamClient()
        worker = NewsWorker(redis_client)
        
        # Run once
        print("Running news worker once...")
        success = await worker.run_once()
        print(f"Run completed with success: {success}")
        
        # Test continuous mode for 30 seconds with short interval
        print("\nRunning news worker continuously for 30 seconds...")
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