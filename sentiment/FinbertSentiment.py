from transformers import pipeline
from .SentimentAnalysisBase import SentimentAnalysisBase
import torch
import pandas as pd

class FinbertSentiment(SentimentAnalysisBase):

    def __init__(self):
        # Use CPU on Streamlit Cloud or fallback if no GPU
        device = 0 if torch.cuda.is_available() else -1

        self._sentiment_analysis = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            device=device
        )
        super().__init__()

    def calc_sentiment_score(self):
        def extract_probs(result):
            # Convert list of dicts into three probability scores
            scores = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
            for r in result:
                scores[r['label']] = r['score']
            return pd.Series([scores['positive'], scores['neutral'], scores['negative']])

        # Run FinBERT sentiment on all news titles
        self.df['sentiment'] = self.df['title'].apply(self._sentiment_analysis)

        # Extract sentiment scores into new columns
        self.df[['positive', 'neutral', 'negative']] = self.df['sentiment'].apply(extract_probs)

        # Compute net sentiment score: positive - negative
        self.df['sentiment_score'] = self.df['positive'] - self.df['negative']
