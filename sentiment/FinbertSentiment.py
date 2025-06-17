from transformers import pipeline
from .SentimentAnalysisBase import SentimentAnalysisBase
import torch
import pandas as pd

class FinbertSentiment(SentimentAnalysisBase):
    def __init__(self):
        # Use CPU fallback if GPU not available (important for Streamlit Cloud)
        device = 0 if torch.cuda.is_available() else -1

        # Load FinBERT pipeline
        self._sentiment_analysis = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            device=device
        )
        super().__init__()

    def calc_sentiment_score(self):
        def extract_probs(result):
            # Convert list of dicts into three scores
            scores = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
            for r in result:
                scores[r['label']] = r['score']
            return pd.Series([scores['positive'], scores['neutral'], scores['negative']])

        # Avoid batch processing: loop one-by-one to avoid torch.meta issues
        results = []
        for title in self.df['title']:
            try:
                result = self._sentiment_analysis(title, truncation=True)
                results.append(result)
            except Exception as e:
                print(f"Error analyzing title: {title}\n{e}")
                # Default fallback to neutral
                results.append([{'label': 'neutral', 'score': 1.0}])

        self.df['sentiment'] = results
        self.df[['positive', 'neutral', 'negative']] = self.df['sentiment'].apply(extract_probs)
        self.df['sentiment_score'] = self.df['positive'] - self.df['negative']
