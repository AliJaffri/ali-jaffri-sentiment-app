from transformers import pipeline
from .SentimentAnalysisBase import SentimentAnalysisBase
import torch
import pandas as pd

class FinbertSentiment(SentimentAnalysisBase):

    def __init__(self):
        device = 0 if torch.cuda.is_available() else -1  # GPU if available, else CPU

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

        titles = self.df['title'].tolist()
        results = self._sentiment_analysis(titles, truncation=True)

        # Put result back into DataFrame
        self.df['sentiment'] = results
        self.df[['positive', 'neutral', 'negative']] = self.df['sentiment'].apply(extract_probs)
        self.df['sentiment_score'] = self.df['positive'] - self.df['negative']
