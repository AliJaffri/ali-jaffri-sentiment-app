import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


class SentimentAnalysisBase:

    def __init__(self):
        self.symbol = None
        self.df = pd.DataFrame()

    def set_symbol(self, symbol):
        self.symbol = symbol

    def set_data(self, df: pd.DataFrame):
        self.df = df

    def calc_sentiment_score(self):
        # This method is meant to be overridden in child classes
        raise NotImplementedError("You must override calc_sentiment_score in a subclass")

    def get_sentiment_scores(self):
        return self.df

    def plot_sentiment(self) -> go.Figure:
        column = 'sentiment_score'

        if column not in self.df.columns:
            raise ValueError(f"'{column}' column not found in the DataFrame")

        df_plot = self.df[self.df[column] != 0].copy()
        df_plot.sort_values(by='Date Time', inplace=True)

        fig = px.bar(
            data_frame=df_plot,
            x='Date Time',
            y=column,
            title=f"{self.symbol} Hourly Sentiment Scores",
            labels={'sentiment_score': 'Sentiment Score'},
            color=column,
            color_continuous_scale='RdYlGn'
        )

        fig.update_layout(
            xaxis_title='Date Time',
            yaxis_title='Sentiment Score',
            title_x=0.5,
            template='plotly_white'
        )

        return fig
