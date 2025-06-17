import os
os.environ["PYTORCH_ENABLE_SDPA"] = "0"  # Critical for disabling SDPA bugs

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TextClassificationPipeline
from sentiment.SentimentAnalysisBase import SentimentAnalysisBase


class FinbertSentiment(SentimentAnalysisBase):
    def __init__(self):
        super().__init__()

        model_name = "ProsusAI/finbert"

        # Explicitly load tokenizer and model (disable meta tensor loading)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # Force model to CPU
        self.model.to(torch.device("cpu"))

        # Use custom pipeline to ensure clean execution
        self._sentiment_analysis = TextClassificationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            device=-1,
            return_all_scores=False,
            truncation=True
        )

    def calc_sentiment_score(self):
        titles = self.df['title'].astype(str).tolist()

        results = self._sentiment_analysis(titles)

        sentiment_scores = []
        sentiments = []

        for result in results:
            label = result['label']
            score = result['score']

            if label == 'positive':
                sentiment_scores.append(score)
                sentiments.append('positive')
            elif label == 'negative':
                sentiment_scores.append(-score)
                sentiments.append('negative')
            else:
                sentiment_scores.append(0)
                sentiments.append('neutral')

        self.df['sentiment_score'] = sentiment_scores
        self.df['sentiment'] = sentiments

    def plot_sentiment(self):
        import plotly.express as px

        df_plot = self.df[self.df['sentiment_score'] != 0]

        fig = px.bar(
            df_plot,
            x='Date Time',
            y='sentiment_score',
            color='sentiment',
            title=f'{self.symbol} Hourly Sentiment Scores',
            labels={'sentiment_score': 'Sentiment Score'},
        )
        fig.update_layout(xaxis_title='Date', yaxis_title='Score', title_x=0.5)
        return fig
