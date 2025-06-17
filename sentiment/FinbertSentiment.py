# === FinbertSentiment.py ===
import os
os.environ["PYTORCH_SDP_DISABLE_FLASH_ATTN"] = "1"  # Disable Flash Attention BEFORE importing torch

import torch
import pandas as pd
from transformers import pipeline
from .SentimentAnalysisBase import SentimentAnalysisBase

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
            # Extract probabilities from model output
            scores = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
            for r in result:
                scores[r['label']] = r['score']
            return pd.Series([scores['positive'], scores['neutral'], scores['negative']])

        titles = self.df['title'].astype(str).tolist()
        results = self._sentiment_analysis(titles, truncation=True)

        self.df['sentiment'] = results
        self.df[['positive', 'neutral', 'negative']] = self.df['sentiment'].apply(extract_probs)
        self.df['sentiment_score'] = self.df['positive'] - self.df['negative']

    def plot_sentiment(self):
        import plotly.express as px
        df_copy = self.df.copy()
        df_copy['Datetime'] = pd.to_datetime(df_copy['Date Time'])
        fig = px.bar(
            df_copy,
            x='Datetime',
            y='sentiment_score',
            color='sentiment_score',
            color_continuous_scale='RdYlGn',
            title='Overall Sentiment Score Over Time'
        )
        fig.update_layout(xaxis_title='Date Time', yaxis_title='Sentiment Score')
        return fig
