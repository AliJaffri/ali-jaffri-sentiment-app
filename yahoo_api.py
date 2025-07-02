from datetime import datetime
import pandas as pd
import pytz
import feedparser
import requests
from config import config

# Define Eastern timezone
EST = pytz.timezone('US/Eastern')

# -------------------------------
# ✅ Get News from Yahoo Finance RSS
# -------------------------------
def get_news(ticker: str, keyword: str = "") -> pd.DataFrame:
    rss_url = f'https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}'
    feed = feedparser.parse(rss_url)

    news_entries = []
    keyword = (keyword or "").lower().strip()

    for entry in feed.entries:
        try:
            title = entry.title
            link = entry.link
            pub_date = datetime.strptime(entry.published, '%a, %d %b %Y %H:%M:%S %z')

            if keyword == "" or keyword in title.lower():
                news_entries.append([
                    pub_date.astimezone(EST),
                    title,
                    entry.get('summary', ''),
                    f'<a href="{link}" target="_blank">{title}</a>'
                ])
        except Exception as e:
            print(f"[ERROR] Skipping entry due to error: {e}")
            continue

    if not news_entries:
        return pd.DataFrame()

    df = pd.DataFrame(news_entries, columns=['Date Time', 'title', 'Description', 'title + link'])
    df.sort_values(by='Date Time', ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# -------------------------------
# ✅ Get Price History via external API
# -------------------------------
def get_price_history(ticker: str, earliest_datetime: pd.Timestamp) -> pd.DataFrame:
    querystring = {
        "symbol": ticker,
        "interval": "5m",
        "diffandsplits": "false"
    }

    try:
        response = requests.get(
            url=config.HISTORY_API_URL,
            headers=config.headers,
            params=querystring,
            timeout=10
        )
        response.raise_for_status()
        json_data = response.json()
    except Exception as e:
        print(f"[ERROR] Failed to fetch price data: {e}")
        return pd.DataFrame(columns=['Date Time', 'Price'])

    if 'body' not in json_data or not isinstance(json_data['body'], dict):
        print(f"[INFO] No valid price data returned for ticker {ticker}")
        return pd.DataFrame(columns=['Date Time', 'Price'])

    price_records = []
    for record in json_data['body'].values():
        try:
            utc_dt = datetime.fromtimestamp(record["date_utc"], tz=pytz.utc)
            est_dt = utc_dt.astimezone(EST)

            if est_dt < earliest_datetime:
                continue

            price = record["open"]
            price_records.append([est_dt, price])
        except Exception as e:
            print(f"[WARN] Skipping malformed price entry: {e}")
            continue

    df = pd.DataFrame(price_records, columns=['Date Time', 'Price'])
    df.sort_values(by='Date Time', inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df
