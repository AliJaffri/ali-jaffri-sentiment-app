from transformers import pipeline
from sentiment.SentimentAnalysisBase import SentimentAnalysisBase  # ✅ absolute import
import torch
import pandas as pd

class FinbertSentiment(SentimentAnalysisBase):
    def __init__(self):
        # Use CPU if no GPU is available
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

        # Run FinBERT sentiment on titles
        titles = self.df['title'].tolist()
        results = self._sentiment_analysis(titles, truncation=True)

        # Parse the results
        scores_list = [extract_probs([r]) for r in results]
        scores_df = pd.DataFrame(scores_list, columns=['positive', 'neutral', 'negative'])

        self.df[['positive', 'neutral', 'negative']] = scores_df
        self.df['sentiment_score'] = self.df['positive'] - self.df['negative']
