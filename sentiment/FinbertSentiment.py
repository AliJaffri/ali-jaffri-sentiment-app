from transformers import pipeline
from .SentimentAnalysisBase import SentimentAnalysisBase
import torch
import pandas as pd

class FinbertSentiment(SentimentAnalysisBase):

    def __init__(self):
        # Ensure compatibility with Streamlit Cloud (CPU-only)
        device = 0 if torch.cuda.is_available() else -1

        self._sentiment_analysis = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            device=device
        )
        super().__init__()

    def calc_sentiment_score(self):
        def extract_probs(result):
            # Convert [{'label': 'neutral', 'score': 0.8}] to separate values
            scores = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
            for r in result:
                scores[r['label']] = r['score']
            return pd.Series([scores['positive'], scores['neutral'], scores['negative']])

        # Run FinBERT on titles
        self.df['sentiment'] = self.df['title'].apply(self._sentiment_analysis)

        # Extract detailed probabilities
        self.df[['score_positive', 'score_neutral', 'score_negative']] = self.df['sentiment'].apply(extract_probs)

        # Compute overall net sentiment score (positive - negative)
        self.df['sentiment_score'] = self.df['score_positive'] - self.df['score_negative']
