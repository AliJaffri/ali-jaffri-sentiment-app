# from datetime import datetime
# import pandas as pd
# import pytz
# import feedparser
# import requests
# from config import config

# # Define Eastern timezone
# EST = pytz.timezone('US/Eastern')

# # -------------------------------
# # ✅ Get News from Yahoo Finance RSS
# # -------------------------------
# def get_news(ticker: str, keyword: str = "") -> pd.DataFrame:
#     rss_url = f'https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}'
#     feed = feedparser.parse(rss_url)

#     news_entries = []
#     keyword = keyword.lower().strip()

#     for entry in feed.entries:
#         try:
#             title = entry.title
#             link = entry.link
#             pub_date = datetime.strptime(entry.published, '%a, %d %b %Y %H:%M:%S %z')

#             if keyword == "" or keyword in title.lower():
#                 news_entries.append([
#                     pub_date.astimezone(EST),
#                     title,
#                     entry.get('summary', ''),
#                     f'<a href="{link}" target="_blank">{title}</a>'
#                 ])
#         except Exception as e:
#             print(f"[ERROR] Skipping entry due to error: {e}")
#             continue

#     if not news_entries:
#         return pd.DataFrame()

#     df = pd.DataFrame(news_entries, columns=['Date Time', 'title', 'Description', 'title + link'])
#     df.sort_values(by='Date Time', ascending=False, inplace=True)
#     df.reset_index(drop=True, inplace=True)
#     return df

# # -------------------------------
# # ✅ Get Price History via external API
# # -------------------------------
# def get_price_history(ticker: str, earliest_datetime: pd.Timestamp) -> pd.DataFrame:
#     querystring = {
#         "symbol": ticker,
#         "interval": "5m",
#         "diffandsplits": "false"
#     }

#     try:
#         response = requests.get(
#             url=config.HISTORY_API_URL,
#             headers=config.headers,
#             params=querystring,
#             timeout=10
#         )
#         response.raise_for_status()
#         json_data = response.json()
#     except Exception as e:
#         print(f"[ERROR] Failed to fetch price data: {e}")
#         return pd.DataFrame(columns=['Date Time', 'Price'])

#     if 'body' not in json_data or not isinstance(json_data['body'], dict):
#         print(f"[INFO] No valid price data returned for ticker {ticker}")
#         return pd.DataFrame(columns=['Date Time', 'Price'])

#     price_records = []
#     for record in json_data['body'].values():
#         try:
#             utc_dt = datetime.fromtimestamp(record["date_utc"], tz=pytz.utc)
#             est_dt = utc_dt.astimezone(EST)

#             if est_dt < earliest_datetime:
#                 continue

#             price = record["open"]
#             price_records.append([est_dt, price])
#         except Exception as e:
#             print(f"[WARN] Skipping malformed price entry: {e}")
#             continue

#     df = pd.DataFrame(price_records, columns=['Date Time', 'Price'])
#     df.sort_values(by='Date Time', inplace=True)
#     df.reset_index(drop=True, inplace=True)

#     return df






import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytz
from flask import Flask, render_template, request
from plotly.utils import PlotlyJSONEncoder

from sentiment.FinbertSentiment import FinbertSentiment
from yahoo_api import get_price_history, get_news  # ✅ Fixed import

EST = pytz.timezone('US/Eastern')
app = Flask(__name__)

sentimentAlgo = FinbertSentiment()

def score_news(news_df: pd.DataFrame) -> pd.DataFrame:
    sentimentAlgo.set_data(news_df)
    sentimentAlgo.calc_sentiment_score()
    return sentimentAlgo.df

def plot_sentiment(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date Time'], y=df['sentiment_score'],
                             mode='lines+markers', name='Sentiment'))
    fig.update_layout(title=f"Sentiment Score Over Time for {ticker}",
                      xaxis_title="Date", yaxis_title="Sentiment Score")
    return fig

def format_headlines(df: pd.DataFrame) -> pd.DataFrame:
    expected_cols = ['Date Time', 'Headline', 'sentiment_score', 'positive', 'neutral', 'negative']
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
        print(f"Available columns: {df.columns.tolist()}")
        return pd.DataFrame()
    return df[expected_cols]

@app.route("/", methods=["GET", "POST"])
def index():
    ticker = request.form.get("ticker", "AAPL")
    news_df = get_news(ticker)
    scored_news_df = score_news(news_df)
    formatted_news_df = format_headlines(scored_news_df)
    chart = plot_sentiment(scored_news_df, ticker)
    chartJSON = json.dumps(chart, cls=PlotlyJSONEncoder)

    return render_template("index.html", chartJSON=chartJSON,
                           tables=[formatted_news_df.to_html(classes='data')],
                           titles=formatted_news_df.columns.values,
                           ticker=ticker)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)


