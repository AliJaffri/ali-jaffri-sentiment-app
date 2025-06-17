import streamlit as st
import pandas as pd
import plotly.express as px
import pytz
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from sentiment.FinbertSentiment import FinbertSentiment
from yahoo_api import get_news, get_price_history

EST = pytz.timezone('US/Eastern')
sentimentAlgo = FinbertSentiment()

def score_news(news_df: pd.DataFrame) -> pd.DataFrame:
    sentimentAlgo.set_data(news_df)
    sentimentAlgo.calc_sentiment_score()
    return sentimentAlgo.df

def get_earliest_date(df: pd.DataFrame) -> pd.Timestamp:
    return df['Date Time'].iloc[-1]

def format_headlines(df: pd.DataFrame) -> pd.DataFrame:
    df['Headline'] = df['title + link']
    return df[['Date Time', 'Headline', 'sentiment_score', 'positive', 'neutral', 'negative']]

def plot_sentiment_and_price(sentiment_df: pd.DataFrame, price_df: pd.DataFrame, ticker: str):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Sentiment Score Over Time", f"{ticker} Price Over Time"))

    fig.add_trace(
        go.Bar(
            x=sentiment_df['Date Time'],
            y=sentiment_df['sentiment_score'],
            name='Sentiment Score',
            marker_color='blue'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=price_df['Date Time'],
            y=price_df['Price'],
            mode='lines',
            name='Price',
            line=dict(color='blue')
        ),
        row=1, col=2
    )

    fig.update_layout(
        title_text=f"{ticker} | Sentiment vs Price",
        showlegend=False,
        height=500,
        width=1000
    )

    return fig

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

        st.subheader("📊 Sentiment and Price Trajectory")
        st.plotly_chart(plot_sentiment_and_price(formatted_news_df, price_history_df, ticker))

        st.subheader("🗞️ News Headlines with Sentiment")
        st.markdown(formatted_news_df.to_html(escape=False, index=False), unsafe_allow_html=True)
