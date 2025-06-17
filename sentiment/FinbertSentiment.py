from transformers import pipeline
from sentiment.SentimentAnalysisBase import SentimentAnalysisBase
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
            scores = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
            for r in result:
                scores[r['label']] = r['score']
            return pd.Series([scores['positive'], scores['neutral'], scores['negative']])

        titles = self.df['title'].astype(str).tolist()
        batch_size = 8  # Safe for most environments

        all_results = []
        for i in range(0, len(titles), batch_size):
            batch = titles[i:i + batch_size]
            try:
                batch_results = self._sentiment_analysis(batch, truncation=True)
                all_results.extend(batch_results)
            except Exception as e:
                print(f"Error in batch {i}-{i + batch_size}: {e}")
                all_results.extend([{'label': 'neutral', 'score': 1.0}] * len(batch))  # fallback

        self.df['sentiment'] = all_results
        sentiment_scores = pd.DataFrame(
            [extract_probs([res]) for res in all_results],
            columns=['positive', 'neutral', 'negative']
        )
        self.df = pd.concat([self.df, sentiment_scores], axis=1)
        self.df['sentiment_score'] = self.df['positive'] - self.df['negative']
