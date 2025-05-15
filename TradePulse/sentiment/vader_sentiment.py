"""
VADER sentiment analysis module for financial text data.

This module uses NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner)
to perform lexicon-based sentiment analysis on financial text.
"""
import os
import sys
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import asyncio

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.logger import setup_logger
import config

logger = setup_logger('vader_sentiment')

class VaderSentimentAnalyzer:
    """VADER-based sentiment analyzer for financial text."""
    
    def __init__(self):
        """Initialize VADER sentiment analyzer."""
        try:
            # Download VADER lexicon if not already present
            try:
                nltk.data.find('sentiment/vader_lexicon.zip')
            except LookupError:
                logger.info("Downloading VADER lexicon...")
                nltk.download('vader_lexicon', quiet=True)
                
            self.analyzer = SentimentIntensityAnalyzer()
            
            # Additional financial terms with sentiment scores
            # These are some common financial terms that might not be captured well by VADER
            financial_terms = {
                # Positive terms
                'bullish': 3.0,
                'outperform': 2.0,
                'buy': 1.5,
                'upgrade': 1.5,
                'beat': 1.0,
                'strong': 1.0,
                'growth': 1.0,
                'profitable': 1.5,
                'upside': 1.5,
                'exceeding': 1.3,
                
                # Negative terms
                'bearish': -3.0,
                'underperform': -2.0,
                'sell': -1.5,
                'downgrade': -1.5,
                'miss': -1.0,
                'weak': -1.0,
                'decline': -1.0,
                'unprofitable': -1.5,
                'downside': -1.5,
                'disappointing': -1.5,
                'bankruptcy': -3.0,
                'debt': -0.5,
                'lawsuit': -1.0,
                
                # Neutral/context-dependent terms (with mild scores)
                'volatility': -0.3,
                'earnings': 0.0,
                'forecast': 0.0
            }
            
            # Add financial terms to VADER lexicon
            for word, score in financial_terms.items():
                self.analyzer.lexicon[word] = score
                
            logger.info("VADER sentiment analyzer initialized with financial terms")
            
        except Exception as e:
            logger.error(f"Error initializing VADER: {e}")
            raise
    
    def analyze_text(self, text):
        """
        Analyze sentiment of a single text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment scores including compound, pos, neg, and neu
        """
        if not text or not isinstance(text, str):
            return {
                'compound': 0.0,
                'pos': 0.0,
                'neg': 0.0,
                'neu': 1.0,
                'sentiment': 'neutral'
            }
        
        try:
            scores = self.analyzer.polarity_scores(text)
            
            # Add a simple sentiment label based on compound score
            compound = scores['compound']
            if compound >= config.VADER_THRESHOLD_POSITIVE:
                scores['sentiment'] = 'positive'
            elif compound <= config.VADER_THRESHOLD_NEGATIVE:
                scores['sentiment'] = 'negative'
            else:
                scores['sentiment'] = 'neutral'
                
            return scores
            
        except Exception as e:
            logger.error(f"Error analyzing text with VADER: {e}")
            return {
                'compound': 0.0,
                'pos': 0.0,
                'neg': 0.0,
                'neu': 1.0,
                'sentiment': 'neutral'
            }
    
    def analyze_df(self, df, text_column, prefix='vader_'):
        """
        Analyze sentiment for texts in a DataFrame column.
        
        Args:
            df (pd.DataFrame): DataFrame containing text data
            text_column (str): Name of column containing text to analyze
            prefix (str): Prefix for new sentiment columns
            
        Returns:
            pd.DataFrame: Input DataFrame with added sentiment columns
        """
        if df.empty or text_column not in df.columns:
            return df
        
        result_df = df.copy()
        
        # Apply sentiment analysis to each text
        result_df[f'{prefix}compound'] = 0.0
        result_df[f'{prefix}positive'] = 0.0
        result_df[f'{prefix}negative'] = 0.0
        result_df[f'{prefix}neutral'] = 0.0
        result_df[f'{prefix}sentiment'] = 'neutral'
        
        for idx, row in df.iterrows():
            text = row.get(text_column, '')
            if text and isinstance(text, str):
                scores = self.analyze_text(text)
                result_df.at[idx, f'{prefix}compound'] = scores['compound']
                result_df.at[idx, f'{prefix}positive'] = scores['pos']
                result_df.at[idx, f'{prefix}negative'] = scores['neg']
                result_df.at[idx, f'{prefix}neutral'] = scores['neu']
                result_df.at[idx, f'{prefix}sentiment'] = scores['sentiment']
        
        return result_df
    
    async def analyze_texts_async(self, texts):
        """
        Analyze sentiment of multiple texts asynchronously.
        
        Args:
            texts (list): List of text strings to analyze
            
        Returns:
            list: List of sentiment score dictionaries
        """
        loop = asyncio.get_event_loop()
        
        # Process analysis in chunks to avoid blocking
        chunk_size = 100
        results = []
        
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i:i + chunk_size]
            
            # Process the chunk in a thread pool
            chunk_results = await loop.run_in_executor(
                None,
                lambda texts_to_process: [self.analyze_text(text) for text in texts_to_process],
                chunk
            )
            
            results.extend(chunk_results)
        
        return results

def get_sentiment_scores(texts):
    """
    Helper function to get sentiment scores for a list of texts.
    
    Args:
        texts (list): List of text strings to analyze
        
    Returns:
        list: List of sentiment score dictionaries
    """
    analyzer = VaderSentimentAnalyzer()
    return [analyzer.analyze_text(text) for text in texts]

async def process_news_sentiment(news_df, text_column='title'):
    """
    Process sentiment for a news DataFrame.
    
    Args:
        news_df (pd.DataFrame): DataFrame with news data
        text_column (str): Column containing text to analyze
        
    Returns:
        pd.DataFrame: News DataFrame with sentiment scores
    """
    if news_df.empty:
        return pd.DataFrame()
    
    analyzer = VaderSentimentAnalyzer()
    return analyzer.analyze_df(news_df, text_column)

if __name__ == "__main__":
    # Example usage
    async def main():
        # Initialize analyzer
        analyzer = VaderSentimentAnalyzer()
        
        # Analyze some financial texts
        texts = [
            "The company reported strong earnings, beating market expectations.",
            "The stock price plummeted after the CEO announced his resignation.",
            "The company's revenue remained stable despite market fluctuations.",
            "Analysts have upgraded the stock from hold to buy.",
            "The company missed earnings targets and forecasts a decline in next quarter's growth."
        ]
        
        # Individual analysis
        for text in texts:
            sentiment = analyzer.analyze_text(text)
            print(f"\nText: {text}")
            print(f"Sentiment: {sentiment['sentiment']}")
            print(f"Compound score: {sentiment['compound']:.2f}")
        
        # Batch analysis
        results = await analyzer.analyze_texts_async(texts)
        print("\nBatch analysis results:")
        for i, result in enumerate(results):
            print(f"{i+1}. Sentiment: {result['sentiment']} ({result['compound']:.2f})")
        
        # DataFrame analysis
        df = pd.DataFrame({'text': texts})
        sentiment_df = analyzer.analyze_df(df, 'text')
        print("\nDataFrame analysis:")
        print(sentiment_df[['text', 'vader_sentiment', 'vader_compound']].head())
    
    asyncio.run(main()) 