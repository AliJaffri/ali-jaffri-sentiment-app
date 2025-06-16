import streamlit as st
import pandas as pd
import plotly.express as px
import pytz

from sentiment.FinbertSentiment import FinbertSentiment
from yahoo_api import get_news, get_price_history

EST = pytz.timezone('US/Eastern')
sentimentAlgo = FinbertSentiment()

def score_news(news_df: pd.DataFrame) -> pd.DataFrame:
    sentimentAlgo.set_data(news_df)
    sentimentAlgo.calc_sentiment_score()
    return sentimentAlgo.df

def plot_sentiment(df: pd.DataFrame):
    return sentimentAlgo.plot_sentiment()

def get_earliest_date(df: pd.DataFrame) -> pd.Timestamp:
    return df['Date Time'].iloc[-1]

def plot_hourly_price(df: pd.DataFrame, ticker: str):
    return px.line(data_frame=df, x='Date Time', y='Price', title=f"{ticker} Price Over Time")

def format_headlines(df: pd.DataFrame) -> pd.DataFrame:
    df['Headline'] = df['title + link']
    return df[['Date Time', 'Headline', 'sentiment_score', 'positive', 'neutral', 'negative']]

# === Streamlit UI ===
st.set_page_config(layout="wide")
st.title("📈 Stock Sentiment Analyzer with FinBERT")

ticker = st.text_input("Enter stock ticker (e.g. AAPL, TSLA):").upper()
keyword = st.text_input("Optional keyword for filtering news (e.g. Tesla, earnings):")

if st.button("Analyze") and ticker:
    with st.spinner("Fetching and analyzing data..."):
        news_df = get_news(ticker=ticker, keyword=keyword if keyword else None)

        if news_df.empty or 'title' not in news_df.columns:
            st.error("❌ No news found or invalid data. Try a popular ticker.")
            st.stop()

        sentimentAlgo.symbol = ticker
        scored_news_df = score_news(news_df)
        formatted_news_df = format_headlines(scored_news_df)

        earliest_datetime = get_earliest_date(news_df)
        price_history_df = get_price_history(ticker, earliest_datetime)

        st.subheader("📰 Sentiment Breakdown")
        st.plotly_chart(plot_sentiment(formatted_news_df))

        st.subheader("📉 Hourly Price Chart")
        st.plotly_chart(plot_hourly_price(price_history_df, ticker))

        st.subheader("🗞️ News Headlines with Sentiment")
        st.markdown(formatted_news_df.to_html(escape=False, index=False), unsafe_allow_html=True)
