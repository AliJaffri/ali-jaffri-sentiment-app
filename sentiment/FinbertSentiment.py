from transformers import pipeline
from .SentimentAnalysisBase import SentimentAnalysisBase
import torch
import pandas as pd

class FinbertSentiment(SentimentAnalysisBase):

    def __init__(self):
        # Use CPU if running on Streamlit Cloud or no GPU is available
        device = 0 if torch.cuda.is_available() else -1

        self._sentiment_analysis = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            device=device
        )
        super().__init__()

    def calc_sentiment_score(self):
        def extract_probs(result):
            # Convert a list of dicts into separate probability scores
            scores = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
            for r in result:
                scores[r['label']] = r['score']
            return pd.Series([scores['positive'], scores['neutral'], scores['negative']])

        # Ensure all titles are strings and collect them
        titles = self.df['title'].astype(str).tolist()

        # Run FinBERT in batch mode
        results = self._sentiment_analysis(titles, truncation=True)

        # Save raw result
        self.df['sentiment'] = results

        # Extract scores into new DataFrame
        sentiment_scores = pd.DataFrame([extract_probs([res]) for res in results],
                                        columns=['positive', 'neutral', 'negative'])

        # Join extracted scores to original DataFrame
        self.df = pd.concat([self.df, sentiment_scores], axis=1)

        # Compute overall sentiment score
        self.df['sentiment_score'] = self.df['positive'] - self.df['negative']
