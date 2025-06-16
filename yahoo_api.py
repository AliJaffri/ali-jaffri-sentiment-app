from datetime import datetime
import pandas as pd
import pytz
import feedparser
import requests
from config import config

EST = pytz.timezone('US/Eastern')
date_format = "%b-%d-%y %H:%M %S"

# -------------------------------
# ✅ Get News using Yahoo Finance RSS
# -------------------------------
def get_news(ticker: str, keyword: str = "") -> pd.DataFrame:
    rss_url = f'https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}'
    feed = feedparser.parse(rss_url)

    data_array = []
    keyword = keyword.lower()

    for entry in feed.entries:
        try:
            title = entry.title
            link = entry.link
            pub_date = datetime.strptime(entry.published, '%a, %d %b %Y %H:%M:%S %z')

            if keyword in title.lower():
                data_array.append([
                    pub_date.astimezone(EST),
                    title,
                    entry.get('summary', ''),
                    f'<a href="{link}" target="_blank">{title}</a>'
                ])
        except Exception as e:
            print(f"Error parsing entry: {e}")
            continue

    if not data_array:
        return pd.DataFrame()

    df = pd.DataFrame(data_array, columns=['Date Time', 'title', 'Description', 'title + link'])
    df.sort_values(by='Date Time', ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# -------------------------------
# ✅ Get Price History
# -------------------------------
def get_price_history(ticker: str, earliest_datetime: pd.Timestamp) -> pd.DataFrame:
    querystring = {
        "symbol": ticker,
        "interval": "5m",
        "diffandsplits": "false"
    }

    response = requests.get(url=config.HISTORY_API_URL, headers=config.headers, params=querystring)
    respose_json = response.json()

    if 'body' not in respose_json:
        print(f"No price data for {ticker}")
        return pd.DataFrame(columns=['Date Time', 'Price'])

    price_history = respose_json['body']
    data_dict = []

    for stock_price in price_history.values():
        date_time_num = stock_price["date_utc"]
        utc_datetime = datetime.fromtimestamp(date_time_num, tz=pytz.utc)
        est_datetime = utc_datetime.astimezone(EST)

        if est_datetime < earliest_datetime:
            continue

        price = stock_price["open"]
        data_dict.append([est_datetime, price])

    df = pd.DataFrame(data_dict, columns=['Date Time', 'Price'])
    df.sort_values(by='Date Time', ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df
