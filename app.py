import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytz
from flask import Flask, render_template, request
from plotly.utils import PlotlyJSONEncoder

from sentiment.FinbertSentiment import FinbertSentiment
from yahoo_data import get_price_history, get_news  # ✅ FIXED

EST = pytz.timezone('US/Eastern')

app = Flask(__name__)

sentimentAlgo = FinbertSentiment()

def score_news(news_df: pd.DataFrame) -> pd.DataFrame:
    sentimentAlgo.set_data(news_df)
    sentimentAlgo.calc_sentiment_score()
    return sentimentAlgo.df

def plot_sentiment(df: pd.DataFrame, ticker: str) -> go.Figure:
    return sentimentAlgo.plot_sentiment()

def get_earliest_date(df: pd.DataFrame) -> pd.Timestamp:
    date = df['Date Time'].iloc[-1]
    py_date = date.to_pydatetime()
    return py_date.replace(tzinfo=EST)

def plot_hourly_price(df, ticker) -> go.Figure:
    fig = px.line(data_frame=df, x='Date Time', y="Price", title=f"{ticker} Price")
    return fig

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    ticker = request.form['ticker'].strip().upper()

    # Step 1: News
    news_df = get_news(ticker)

    # Step 2: Sentiment
    scored_news_df = score_news(news_df)

    # Step 3: Bar Plot of Sentiment
    fig_bar_sentiment = plot_sentiment(scored_news_df, ticker)
    graph_sentiment = json.dumps(fig_bar_sentiment, cls=PlotlyJSONEncoder)

    # Step 4: Earliest news time
    earliest_datetime = get_earliest_date(news_df)

    # Step 5: Price history
    price_history_df = get_price_history(ticker, earliest_datetime)

    # Step 6: Line plot of price
    fig_line_price_history = plot_hourly_price(price_history_df, ticker)
    graph_price = json.dumps(fig_line_price_history, cls=PlotlyJSONEncoder)

    # Step 7: Clickable headlines
    scored_news_df = convert_headline_to_link(scored_news_df)

    # Step 8: Render page
    return render_template(
        'analysis.html',
        ticker=ticker,
        graph_price=graph_price,
        graph_sentiment=graph_sentiment,
        table=scored_news_df.to_html(classes='mystyle', render_links=True, escape=False)
    )

def convert_headline_to_link(df: pd.DataFrame) -> pd.DataFrame:
    df.insert(2, 'Headline', df['title + link'])
    df.drop(columns=['sentiment', 'title + link', 'title'], inplace=True, axis=1)
    return df

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)


