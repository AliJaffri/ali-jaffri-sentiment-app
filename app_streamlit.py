import streamlit as st
import pandas as pd
import plotly.express as px
import pytz
import json
from plotly.utils import PlotlyJSONEncoder

from sentiment.FinbertSentiment import FinbertSentiment
from yahoo_api import API

EST = pytz.timezone('US/Eastern')

sentimentAlgo = FinbertSentiment()

def get_price_history(ticker: str, earliest_datetime: pd.Timestamp) -> pd.DataFrame:
    return API.get_price_history(ticker, earliest_datetime)

def get_news(ticker) -> pd.DataFrame:
    sentimentAlgo.set_symbol(ticker)
    return API.get_news(ticker)

def score_news(news_df: pd.DataFrame) -> pd.DataFrame:
    sentimentAlgo.set_data(news_df)
    sentimentAlgo.calc_sentiment_score()
    return sentimentAlgo.df

def plot_sentiment(df: pd.DataFrame):
    return sentimentAlgo.plot_sentiment()

def get_earliest_date(df: pd.DataFrame) -> pd.Timestamp:
    date = df['Date Time'].iloc[-1]
    py_date = date.to_pydatetime()
    return EST.localize(py_date)

def plot_hourly_price(df, ticker):
    return px.line(data_frame=df, x='Date Time', y="Price", title=f"{ticker} Price")

def convert_headline_to_link(df: pd.DataFrame) -> pd.DataFrame:
    df['Headline'] = df['title + link'].apply(lambda x: f'<a href="{x[1]}" target="_blank">{x[0]}</a>')
    df.drop(columns=['sentiment', 'title + link', 'title'], inplace=True, axis=1)
    return df

# === Streamlit UI ===

st.set_page_config(layout="wide")
st.title("📈 Stock Sentiment Analyzer with FinBERT")

ticker = st.text_input("Enter stock ticker (e.g. AAPL, TSLA):").upper()

if st.button("Analyze") and ticker:
    with st.spinner("Fetching and analyzing data..."):
        news_df = get_news(ticker)

        # Check if news is empty or missing required columns
        if news_df.empty or 'title' not in news_df.columns:
            st.error("❌ No news found or invalid data for this ticker. Try a popular one like AAPL or TSLA.")
            st.stop()

        scored_news_df = score_news(news_df)


        earliest_datetime = get_earliest_date(news_df)
        price_history_df = get_price_history(ticker, earliest_datetime)

        st.subheader("📰 Sentiment Breakdown")
        st.plotly_chart(plot_sentiment(scored_news_df))

        st.subheader("📉 Hourly Price Chart")
        st.plotly_chart(plot_hourly_price(price_history_df, ticker))

        st.subheader("🗞️ News Headlines with Sentiment")
        st.markdown(scored_news_df.to_html(escape=False, index=False), unsafe_allow_html=True)
