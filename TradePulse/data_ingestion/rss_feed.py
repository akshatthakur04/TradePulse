"""
RSS Feed client for gathering financial news headlines.

This module fetches and processes news from financial news sources
using RSS feeds via the feedparser library.
"""
import os
import sys
import asyncio
import feedparser
import pandas as pd
from datetime import datetime, timedelta
import aiohttp
from newspaper import Article
import time
import re

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.logger import setup_logger
import config

logger = setup_logger('rss_feed')

# Financial news RSS feed URLs
FINANCIAL_RSS_FEEDS = {
    "reuters_business": "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best",
    "reuters_markets": "https://www.reutersagency.com/feed/?best-topics=markets&post_type=best",
    "seeking_alpha_market_news": "https://seekingalpha.com/market_currents.xml",
    "yahoo_finance": "https://finance.yahoo.com/news/rssindex",
    "cnbc_investing": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839069",
    "investing_com_news": "https://www.investing.com/rss/news.rss"
}

class RSSNewsFeed:
    """Client for fetching financial news from RSS feeds."""

    def __init__(self, feeds=None):
        """
        Initialize the RSS news feed client.
        
        Args:
            feeds (dict, optional): Dict of feed names to URLs. If not provided,
                                  default feeds will be used.
        """
        self.feeds = feeds or FINANCIAL_RSS_FEEDS
        self.user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

    async def fetch_all_feeds(self):
        """
        Fetch all configured RSS feeds in parallel.
        
        Returns:
            pd.DataFrame: Combined dataframe with all news items
        """
        results = []
        tasks = []
        
        for feed_name, feed_url in self.feeds.items():
            task = asyncio.create_task(self._fetch_feed(feed_name, feed_url))
            tasks.append(task)
        
        feed_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in feed_results:
            if isinstance(result, Exception):
                logger.error(f"Error fetching feed: {result}")
            elif isinstance(result, pd.DataFrame) and not result.empty:
                results.append(result)
        
        if not results:
            return pd.DataFrame()
        
        return pd.concat(results, ignore_index=True)

    async def _fetch_feed(self, feed_name, feed_url):
        """
        Fetch a single RSS feed and process its entries.
        
        Args:
            feed_name (str): Name of the feed source
            feed_url (str): URL of the RSS feed
            
        Returns:
            pd.DataFrame: Processed news items from the feed
        """
        try:
            # Using feedparser as a synchronous call since it doesn't support async natively
            logger.debug(f"Fetching feed: {feed_name} from {feed_url}")
            feed = feedparser.parse(feed_url, agent=self.user_agent)
            
            if not feed.entries:
                logger.warning(f"No entries found for feed {feed_name}")
                return pd.DataFrame()
            
            data = []
            for entry in feed.entries:
                # Extract basic info from feed entry
                item = {
                    'title': entry.get('title', ''),
                    'link': entry.get('link', ''),
                    'published': entry.get('published', entry.get('updated', '')),
                    'summary': entry.get('summary', ''),
                    'source': feed_name
                }
                
                # Convert published date to datetime if possible
                if item['published']:
                    try:
                        dt = pd.to_datetime(item['published'])
                        item['published'] = dt.strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        # Keep as string if parsing fails
                        pass
                
                data.append(item)
            
            df = pd.DataFrame(data)
            logger.info(f"Fetched {len(df)} news items from {feed_name}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching feed {feed_name}: {e}")
            return pd.DataFrame()
    
    def extract_stock_symbols(self, text):
        """
        Extract potential stock ticker symbols from text.
        
        Args:
            text (str): Text to scan for stock symbols
            
        Returns:
            list: Extracted potential stock symbols
        """
        # Simple regex for common ticker patterns
        # Looks for standalone capital letters that may be tickers
        # This is basic and will have false positives
        pattern = r'\b[A-Z]{1,5}\b'
        matches = re.findall(pattern, text)
        
        # Filter out common English words and abbreviations
        common_words = {'A', 'I', 'AM', 'PM', 'CEO', 'CFO', 'CTO', 'COO', 'IPO', 'GDP',
                        'USD', 'EU', 'UK', 'US', 'IT', 'AI', 'FBI', 'SEC', 'FED', 'THE'}
        return [m for m in matches if m not in common_words]

    async def fetch_article_content(self, url):
        """
        Fetch and extract the content of a news article.
        
        Args:
            url (str): URL of the article to fetch
            
        Returns:
            dict: Article data including title, text, and metadata
        """
        try:
            article = Article(url)
            
            # Manual download to integrate with asyncio
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers={'User-Agent': self.user_agent}) as response:
                    if response.status != 200:
                        logger.error(f"Error fetching article {url}: {response.status}")
                        return None
                    
                    html = await response.text()
            
            article.set_html(html)
            article.parse()
            article.nlp()  # Extract keywords, summary
            
            return {
                'title': article.title,
                'text': article.text,
                'summary': article.summary,
                'keywords': article.keywords,
                'publish_date': article.publish_date,
                'url': url
            }
        
        except Exception as e:
            logger.error(f"Error processing article {url}: {e}")
            return None

    async def enrich_news_with_content(self, news_df, max_articles=10):
        """
        Enrich news dataframe with article content for a limited number of articles.
        
        Args:
            news_df (pd.DataFrame): Dataframe with news items
            max_articles (int): Maximum number of articles to fetch full content for
            
        Returns:
            pd.DataFrame: Enriched news dataframe
        """
        if news_df.empty or 'link' not in news_df.columns:
            return news_df
        
        # Limit to most recent articles
        recent_df = news_df.sort_values('published', ascending=False).head(max_articles)
        
        tasks = []
        for idx, row in recent_df.iterrows():
            task = self.fetch_article_content(row['link'])
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Add content to the dataframe
        for i, content in enumerate(results):
            if content:
                idx = recent_df.index[i]
                news_df.loc[idx, 'full_text'] = content.get('text', '')
                news_df.loc[idx, 'keywords'] = ','.join(content.get('keywords', []))
        
        # Extract potential stock symbols
        news_df['symbols'] = news_df.apply(
            lambda row: self.extract_stock_symbols(row['title'] + ' ' + row.get('full_text', '')), 
            axis=1
        )
        
        return news_df

async def get_recent_financial_news(max_articles=10):
    """
    Fetch recent financial news from all configured sources.
    
    Args:
        max_articles (int): Maximum number of articles to fetch full content for
        
    Returns:
        pd.DataFrame: News dataframe with enriched content
    """
    client = RSSNewsFeed()
    news_df = await client.fetch_all_feeds()
    
    if not news_df.empty:
        # Enrich with content for a limited number of articles
        news_df = await client.enrich_news_with_content(news_df, max_articles)
    
    return news_df

if __name__ == "__main__":
    # Example usage
    async def main():
        # Get recent news
        news = await get_recent_financial_news(max_articles=3)
        
        if not news.empty:
            # Print news headlines
            print(f"Retrieved {len(news)} news items")
            for _, item in news.iterrows():
                print(f"\nHeadline: {item['title']}")
                print(f"Source: {item['source']}")
                print(f"Published: {item['published']}")
                print(f"Link: {item['link']}")
                if 'symbols' in item and item['symbols']:
                    print(f"Potential stock symbols: {item['symbols']}")
    
    asyncio.run(main()) 