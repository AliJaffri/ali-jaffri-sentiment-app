import os
os.environ["PYTORCH_SDP_DISABLE_FLASH_ATTN"] = "1"

from transformers import pipeline
from .SentimentAnalysisBase import SentimentAnalysisBase
import torch
import pandas as pd

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
                scores[r['label']] = r['score']
            return pd.Series([scores['positive'], scores['neutral'], scores['negative']])

        titles = self.df['title'].tolist()
        results = self._sentiment_analysis(titles, truncation=True)
        sentiment_df = pd.DataFrame(results)
        self.df['sentiment'] = results
        self.df[['positive', 'neutral', 'negative']] = self.df['sentiment'].apply(lambda r: pd.Series({
            'positive': r[0]['score'] if r[0]['label'] == 'positive' else 0.0,
            'neutral': r[0]['score'] if r[0]['label'] == 'neutral' else 0.0,
            'negative': r[0]['score'] if r[0]['label'] == 'negative' else 0.0,
        }))
        self.df['sentiment_score'] = self.df['positive'] - self.df['negative']
