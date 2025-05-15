"""
FinBERT transformer-based sentiment analysis for financial text.

This module uses the FinBERT model from the transformers library, which is
a BERT model fine-tuned on financial text for sentiment analysis.
"""
import os
import sys
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import asyncio
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.logger import setup_logger
import config

logger = setup_logger('finbert_sentiment')

class FinBERTSentimentAnalyzer:
    """FinBERT-based sentiment analyzer for financial text."""
    
    def __init__(self, model_name="ProsusAI/finbert"):
        """
        Initialize FinBERT sentiment analyzer.
        
        Args:
            model_name (str): Model name or path from Hugging Face hub
        """
        try:
            # Check for CUDA availability
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
            
            # Load model and tokenizer
            logger.info(f"Loading FinBERT model from {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            
            # Set evaluation mode
            self.model.eval()
            
            # Define label mapping (specific to FinBERT)
            self.labels = ["negative", "neutral", "positive"]
            
            logger.info("FinBERT sentiment analyzer initialized")
        
        except Exception as e:
            logger.error(f"Error initializing FinBERT analyzer: {e}")
            raise
    
    def _batch_tokenize(self, texts, max_length=512):
        """
        Tokenize a batch of texts.
        
        Args:
            texts (list): List of text strings
            max_length (int): Maximum sequence length
            
        Returns:
            dict: Tokenized inputs
        """
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(self.device)
    
    def analyze_batch(self, texts, batch_size=8):
        """
        Analyze sentiment for a batch of texts.
        
        Args:
            texts (list): List of text strings
            batch_size (int): Batch size for inference
            
        Returns:
            list: List of sentiment dictionaries
        """
        if not texts:
            return []
        
        results = []
        
        # Process in batches to avoid OOM errors
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Skip empty texts
            batch_texts = [text if text and isinstance(text, str) else "" for text in batch_texts]
            
            # Tokenize
            inputs = self._batch_tokenize(batch_texts)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get probabilities with softmax
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
            
            # Process each prediction
            for text, prob in zip(batch_texts, probs):
                label_id = np.argmax(prob)
                sentiment = self.labels[label_id]
                
                result = {
                    'sentiment': sentiment,
                    'confidence': float(prob[label_id]),
                    'probabilities': {
                        'negative': float(prob[0]),
                        'neutral': float(prob[1]),
                        'positive': float(prob[2])
                    },
                    'compound': float(prob[2] - prob[0])  # Simplified compound: positive - negative
                }
                
                results.append(result)
        
        return results
    
    async def analyze_batch_async(self, texts, batch_size=8):
        """
        Analyze sentiment asynchronously.
        
        Args:
            texts (list): List of text strings
            batch_size (int): Batch size for inference
            
        Returns:
            list: List of sentiment dictionaries
        """
        loop = asyncio.get_event_loop()
        
        # Process in chunks to avoid blocking
        max_chunk_size = 64
        results = []
        
        for i in range(0, len(texts), max_chunk_size):
            chunk = texts[i:i + max_chunk_size]
            
            # Process the chunk in a thread pool
            chunk_results = await loop.run_in_executor(
                None,
                lambda chunk_texts: self.analyze_batch(chunk_texts, batch_size),
                chunk
            )
            
            results.extend(chunk_results)
        
        return results
    
    def analyze_df(self, df, text_column, prefix='finbert_'):
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
        
        # Prepare texts
        texts = df[text_column].tolist()
        
        # Analyze sentiment
        results = self.analyze_batch(texts)
        
        # Add results to DataFrame
        result_df = df.copy()
        result_df[f'{prefix}sentiment'] = [r['sentiment'] for r in results]
        result_df[f'{prefix}confidence'] = [r['confidence'] for r in results]
        result_df[f'{prefix}negative'] = [r['probabilities']['negative'] for r in results]
        result_df[f'{prefix}neutral'] = [r['probabilities']['neutral'] for r in results]
        result_df[f'{prefix}positive'] = [r['probabilities']['positive'] for r in results]
        result_df[f'{prefix}compound'] = [r['compound'] for r in results]
        
        return result_df

async def process_news_sentiment(news_df, text_column='title'):
    """
    Process sentiment for a news DataFrame using FinBERT.
    
    Args:
        news_df (pd.DataFrame): DataFrame with news data
        text_column (str): Column containing text to analyze
        
    Returns:
        pd.DataFrame: News DataFrame with sentiment scores
    """
    if news_df.empty:
        return pd.DataFrame()
    
    analyzer = FinBERTSentimentAnalyzer()
    
    # Process in chunks to avoid memory issues
    texts = news_df[text_column].tolist()
    results = await analyzer.analyze_batch_async(texts)
    
    # Add results to DataFrame
    for i, result in enumerate(results):
        news_df.loc[i, 'finbert_sentiment'] = result['sentiment']
        news_df.loc[i, 'finbert_confidence'] = result['confidence']
        news_df.loc[i, 'finbert_compound'] = result['compound']
    
    return news_df

if __name__ == "__main__":
    # Example usage
    async def main():
        # Initialize analyzer
        analyzer = FinBERTSentimentAnalyzer()
        
        # Analyze some financial texts
        texts = [
            "The company reported strong earnings, beating market expectations.",
            "The stock price plummeted after the CEO announced his resignation.",
            "The company's revenue remained stable despite market fluctuations.",
            "Analysts have upgraded the stock from hold to buy.",
            "The company missed earnings targets and forecasts a decline in next quarter's growth."
        ]
        
        # Batch analysis
        results = analyzer.analyze_batch(texts)
        print("FinBERT Sentiment Analysis Results:")
        for i, (text, result) in enumerate(zip(texts, results)):
            print(f"\n{i+1}. Text: {text}")
            print(f"   Sentiment: {result['sentiment']}")
            print(f"   Confidence: {result['confidence']:.3f}")
            print(f"   Compound: {result['compound']:.3f}")
            print(f"   Probabilities: " + 
                  f"Positive={result['probabilities']['positive']:.3f}, " +
                  f"Neutral={result['probabilities']['neutral']:.3f}, " +
                  f"Negative={result['probabilities']['negative']:.3f}")
        
        # Async analysis
        async_results = await analyzer.analyze_batch_async(texts)
        print("\nAsync Analysis (should match above):")
        for i, result in enumerate(async_results):
            print(f"{i+1}. {result['sentiment']} (confidence: {result['confidence']:.3f})")
        
        # DataFrame analysis
        df = pd.DataFrame({'text': texts})
        sentiment_df = analyzer.analyze_df(df, 'text')
        print("\nDataFrame analysis:")
        print(sentiment_df[['text', 'finbert_sentiment', 'finbert_compound']].head())
    
    asyncio.run(main()) 