"""
Social media streams client for financial discussions.

This module fetches posts from platforms like StockTwits and Reddit
to gather social sentiment on financial instruments.
"""
import os
import sys
import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timedelta
import praw
import json

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.logger import setup_logger
import config

logger = setup_logger('social_streams')

class StockTwitsClient:
    """Client for fetching stock-related posts from StockTwits."""
    
    def __init__(self, api_key=None):
        """
        Initialize StockTwits client.
        
        Args:
            api_key (str, optional): StockTwits API key. If not provided,
                                   it will be read from config.
        """
        self.api_key = api_key or config.STOCKTWITS_API_KEY
        self.base_url = "https://api.stocktwits.com/api/2"
        
        # Note: StockTwits has a free tier that doesn't require authentication
        # for some endpoints, but for better access, use the API key if available
        
    async def _make_request(self, endpoint, params=None):
        """
        Make a request to StockTwits API.
        
        Args:
            endpoint (str): API endpoint to query
            params (dict, optional): Additional query parameters
            
        Returns:
            dict: JSON response or None on error
        """
        url = f"{self.base_url}/{endpoint}"
        headers = {}
        
        if self.api_key:
            headers['Authorization'] = f"OAuth {self.api_key}"
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        logger.error(f"Error with StockTwits API: {response.status} - {error_text}")
                        return None
            except Exception as e:
                logger.error(f"Exception while calling StockTwits API: {e}")
                return None
    
    async def get_symbol_stream(self, symbol):
        """
        Get the message stream for a specific symbol.
        
        Args:
            symbol (str): Stock ticker symbol
            
        Returns:
            pd.DataFrame: DataFrame with messages, or empty DataFrame on error
        """
        endpoint = f"streams/symbol/{symbol}.json"
        data = await self._make_request(endpoint)
        
        if not data or 'messages' not in data:
            logger.warning(f"No StockTwits messages found for {symbol}")
            return pd.DataFrame()
        
        try:
            messages = []
            for msg in data['messages']:
                # Extract relevant information
                message = {
                    'id': msg.get('id'),
                    'body': msg.get('body', ''),
                    'created_at': msg.get('created_at', ''),
                    'symbols': [s.get('symbol') for s in msg.get('symbols', [])],
                    'user_id': msg.get('user', {}).get('id'),
                    'username': msg.get('user', {}).get('username'),
                    'followers': msg.get('user', {}).get('followers', 0),
                    'source': 'stocktwits'
                }
                
                # Add sentiment if available
                if 'entities' in msg and 'sentiment' in msg['entities']:
                    message['sentiment'] = msg['entities']['sentiment'].get('basic', '')
                
                messages.append(message)
            
            df = pd.DataFrame(messages)
            if not df.empty:
                # Convert created_at to datetime
                df['created_at'] = pd.to_datetime(df['created_at'])
                
                # Convert symbols list to string for easier processing
                df['symbols'] = df['symbols'].apply(lambda x: ','.join(x) if x else '')
            
            return df
        
        except Exception as e:
            logger.error(f"Error processing StockTwits data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def get_trending_symbols(self):
        """
        Get trending symbols on StockTwits.
        
        Returns:
            pd.DataFrame: DataFrame with trending symbols
        """
        endpoint = "trending/symbols.json"
        data = await self._make_request(endpoint)
        
        if not data or 'symbols' not in data:
            logger.warning("No trending symbols found")
            return pd.DataFrame()
        
        try:
            symbols_data = []
            for symbol in data['symbols']:
                symbols_data.append({
                    'symbol': symbol.get('symbol', ''),
                    'title': symbol.get('title', ''),
                    'watchlist_count': symbol.get('watchlist_count', 0),
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
            
            return pd.DataFrame(symbols_data)
        
        except Exception as e:
            logger.error(f"Error processing trending symbols: {e}")
            return pd.DataFrame()


class RedditClient:
    """Client for fetching stock-related posts from Reddit."""
    
    def __init__(self, client_id=None, client_secret=None, user_agent=None):
        """
        Initialize Reddit client.
        
        Args:
            client_id (str, optional): Reddit API client ID. If not provided,
                                    it will be read from config.
            client_secret (str, optional): Reddit API client secret. If not provided,
                                        it will be read from config.
            user_agent (str, optional): User agent to identify your app. If not provided,
                                      it will be read from config.
        """
        self.client_id = client_id or config.REDDIT_CLIENT_ID
        self.client_secret = client_secret or config.REDDIT_CLIENT_SECRET
        self.user_agent = user_agent or config.REDDIT_USER_AGENT
        
        self.subreddits = [
            'wallstreetbets', 
            'investing', 
            'stocks', 
            'stockmarket',
            'options',
            'pennystocks'
        ]
        
        # Initialize PRAW instance if credentials available
        if self.client_id and self.client_secret and self.user_agent:
            try:
                self.reddit = praw.Reddit(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    user_agent=self.user_agent
                )
                logger.info("Reddit client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Reddit client: {e}")
                self.reddit = None
        else:
            logger.warning("Reddit credentials not provided, functionality will be limited")
            self.reddit = None
    
    async def get_hot_posts(self, subreddit_name, limit=25):
        """
        Get hot posts from a subreddit.
        
        Args:
            subreddit_name (str): Name of the subreddit
            limit (int): Maximum number of posts to fetch
            
        Returns:
            pd.DataFrame: DataFrame with posts
        """
        if not self.reddit:
            logger.error("Reddit client not initialized")
            return pd.DataFrame()
        
        try:
            # Using executor to run synchronous PRAW code asynchronously
            loop = asyncio.get_event_loop()
            
            def fetch_posts():
                subreddit = self.reddit.subreddit(subreddit_name)
                posts_data = []
                
                for post in subreddit.hot(limit=limit):
                    posts_data.append({
                        'id': post.id,
                        'title': post.title,
                        'score': post.score,
                        'num_comments': post.num_comments,
                        'created_utc': datetime.fromtimestamp(post.created_utc),
                        'url': post.url,
                        'selftext': post.selftext if hasattr(post, 'selftext') else '',
                        'upvote_ratio': post.upvote_ratio if hasattr(post, 'upvote_ratio') else None,
                        'subreddit': subreddit_name,
                        'source': 'reddit'
                    })
                
                return posts_data
            
            posts = await loop.run_in_executor(None, fetch_posts)
            
            if not posts:
                logger.warning(f"No posts found in r/{subreddit_name}")
                return pd.DataFrame()
            
            df = pd.DataFrame(posts)
            return df
            
        except Exception as e:
            logger.error(f"Error fetching posts from r/{subreddit_name}: {e}")
            return pd.DataFrame()
    
    def extract_ticker_symbols(self, text):
        """
        Extract potential ticker symbols from text using $ prefix common on Reddit.
        
        Args:
            text (str): Text to scan for ticker symbols
            
        Returns:
            list: Extracted ticker symbols
        """
        import re
        # Look for $TICKER pattern (common on Reddit)
        pattern = r'\$([A-Z]{1,5})\b'
        matches = re.findall(pattern, text)
        
        # Remove common false positives
        common_words = {'A', 'I', 'AM', 'PM', 'CEO', 'CFO', 'CTO', 'COO', 'IPO'}
        return [m for m in matches if m not in common_words]
    
    async def search_for_symbol(self, symbol, subreddit_name=None, limit=25):
        """
        Search for posts containing a specific symbol.
        
        Args:
            symbol (str): Stock ticker symbol to search for
            subreddit_name (str, optional): Specific subreddit to search in
            limit (int): Maximum number of search results
            
        Returns:
            pd.DataFrame: DataFrame with search results
        """
        if not self.reddit:
            logger.error("Reddit client not initialized")
            return pd.DataFrame()
        
        try:
            loop = asyncio.get_event_loop()
            
            def search_posts():
                # If subreddit provided, search only that one
                if subreddit_name:
                    subreddit = self.reddit.subreddit(subreddit_name)
                else:
                    # Otherwise search across multiple finance subreddits
                    subreddit = self.reddit.subreddit('+'.join(self.subreddits))
                
                # Search for both the symbol and $symbol format
                query = f"{symbol} OR ${symbol}"
                search_results = []
                
                for post in subreddit.search(query, sort='new', time_filter='week', limit=limit):
                    search_results.append({
                        'id': post.id,
                        'title': post.title,
                        'score': post.score,
                        'num_comments': post.num_comments,
                        'created_utc': datetime.fromtimestamp(post.created_utc),
                        'url': post.url,
                        'selftext': post.selftext if hasattr(post, 'selftext') else '',
                        'subreddit': post.subreddit.display_name,
                        'source': 'reddit'
                    })
                
                return search_results
            
            results = await loop.run_in_executor(None, search_posts)
            
            if not results:
                logger.warning(f"No posts found for symbol {symbol}")
                return pd.DataFrame()
            
            df = pd.DataFrame(results)
            return df
            
        except Exception as e:
            logger.error(f"Error searching for symbol {symbol}: {e}")
            return pd.DataFrame()


async def get_social_sentiment_for_symbols(symbols):
    """
    Get combined social media sentiment for a list of symbols.
    
    Args:
        symbols (list): List of stock ticker symbols
        
    Returns:
        pd.DataFrame: Combined dataframe with social media posts
    """
    results = []
    
    # StockTwits data
    stocktwits = StockTwitsClient()
    for symbol in symbols:
        df = await stocktwits.get_symbol_stream(symbol)
        if not df.empty:
            results.append(df)
    
    # Reddit data
    reddit = RedditClient()
    if reddit.reddit:  # Only if Reddit client was successfully initialized
        for symbol in symbols:
            df = await reddit.search_for_symbol(symbol)
            if not df.empty:
                results.append(df)
    
    if not results:
        return pd.DataFrame()
    
    # Combine results
    combined_df = pd.concat(results, ignore_index=True)
    
    # Add timestamp for easier aggregation
    combined_df['timestamp'] = datetime.now()
    
    return combined_df

if __name__ == "__main__":
    # Example usage
    async def main():
        # Get social sentiment for selected symbols
        symbols = ["AAPL", "TSLA", "GME"]
        social_df = await get_social_sentiment_for_symbols(symbols)
        
        if not social_df.empty:
            print(f"Retrieved {len(social_df)} social media posts")
            
            # Show sample of StockTwits data
            stocktwits_posts = social_df[social_df['source'] == 'stocktwits']
            if not stocktwits_posts.empty:
                print("\nSample StockTwits posts:")
                for _, post in stocktwits_posts.head(2).iterrows():
                    print(f"User: {post.get('username')} - {post.get('body')}")
                    if 'sentiment' in post:
                        print(f"Sentiment: {post.get('sentiment')}")
            
            # Show sample of Reddit data
            reddit_posts = social_df[social_df['source'] == 'reddit']
            if not reddit_posts.empty:
                print("\nSample Reddit posts:")
                for _, post in reddit_posts.head(2).iterrows():
                    print(f"r/{post.get('subreddit')} - {post.get('title')}")
                    print(f"Score: {post.get('score')} | Comments: {post.get('num_comments')}")
        
        # Get trending symbols on StockTwits
        stocktwits_client = StockTwitsClient()
        trending = await stocktwits_client.get_trending_symbols()
        
        if not trending.empty:
            print("\nTrending symbols on StockTwits:")
            print(trending.head(5)[['symbol', 'watchlist_count']])
    
    asyncio.run(main()) 