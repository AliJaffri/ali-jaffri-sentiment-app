# ===============================
# File: sentiment/FinbertSentiment.py
# ===============================
import os
import torch
import pandas as pd
from transformers import pipeline
from .SentimentAnalysisBase import SentimentAnalysisBase

# Disable SDPA Flash Attention for compatibility
os.environ["PYTORCH_SDP_DISABLE_FLASH_ATTN"] = "1"

class FinbertSentiment(SentimentAnalysisBase):

    def __init__(self):
        device = 0 if torch.cuda.is_available() else -1
        self._sentiment_analysis = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            device=device
        )
        super().__init__()

    def calc_sentiment_score(self):
        def extract_probs(result):
            scores = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
            for r in result:
                scores[r['label'].lower()] = r['score']
            return pd.Series([scores['positive'], scores['neutral'], scores['negative']])

        # Batch process titles
        titles = self.df['title'].tolist()
        results = self._sentiment_analysis(titles, truncation=True)

        self.df['sentiment'] = results
        self.df[['positive', 'neutral', 'negative']] = self.df['sentiment'].apply(lambda x: pd.Series({
            'positive': x['score'] if x['label'].lower() == 'positive' else 0.0,
            'neutral': x['score'] if x['label'].lower() == 'neutral' else 0.0,
            'negative': x['score'] if x['label'].lower() == 'negative' else 0.0
        }))

        # Calculate net sentiment
        self.df['sentiment_score'] = self.df['positive'] - self.df['negative']
