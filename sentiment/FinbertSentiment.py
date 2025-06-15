from transformers import pipeline
from .SentimentAnalysisBase import SentimentAnalysisBase
import torch

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
        self.df['sentiment'] = self.df['title'].apply(self._sentiment_analysis)
        self.df['sentiment_score'] = self.df['sentiment'].apply(
            lambda x: {x[0]['label'] == 'negative': -1, x[0]['label'] == 'positive': 1}.get(True, 0) * x[0]['score']
        )
