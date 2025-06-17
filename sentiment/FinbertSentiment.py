from transformers import pipeline
from .SentimentAnalysisBase import SentimentAnalysisBase
import torch
import pandas as pd

class FinbertSentiment(SentimentAnalysisBase):
    def __init__(self):
        # Automatically choose GPU if available, otherwise fallback to CPU
        device = 0 if torch.cuda.is_available() else -1

        # Load FinBERT with all scores enabled
        self._sentiment_analysis = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            return_all_scores=True,  # ✅ get full distribution
            device=device
        )
        super().__init__()

    def calc_sentiment_score(self):
        def extract_probs(result):
            """
            Convert a list of dictionaries like:
            [{'label': 'positive', 'score': 0.1}, {'label': 'neutral', 'score': 0.8}, {'label': 'negative', 'score': 0.1}]
            into a Series: [0.1, 0.8, 0.1]
            """
            scores = {entry['label']: entry['score'] for entry in result}
            return pd.Series([
                scores.get('positive', 0.0),
                scores.get('neutral', 0.0),
                scores.get('negative', 0.0)
            ])

        results = []
        for title in self.df['title']:
            try:
                result = self._sentiment_analysis(title, truncation=True)
                results.append(result[0])  # extract top list (1 element with 3 labels)
            except Exception as e:
                print(f"[ERROR] Sentiment failed for title: {title}\n{e}")
                results.append([
                    {'label': 'positive', 'score': 0.0},
                    {'label': 'neutral', 'score': 1.0},
                    {'label': 'negative', 'score': 0.0}
                ])

        self.df['sentiment'] = results
        self.df[['positive', 'neutral', 'negative']] = self.df['sentiment'].apply(extract_probs)
        self.df['sentiment_score'] = self.df['positive'] - self.df['negative']
