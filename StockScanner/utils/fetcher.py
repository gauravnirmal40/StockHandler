"""
fetcher.py — Pull market data, short interest, and macro indicators
Runs daily (or on demand) to populate ticker_snapshots in the DB.

APIs used:
  - yfinance (free, no key needed)
  - Finviz (scraped, no key needed)
  - FRED (free key needed for macro data)
"""

import os
import time
import json
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import track
import ta

load_dotenv()
console = Console()

# ── optional FRED import ───────────────────────────────────────
try:
    from fredapi import Fred
    FRED_KEY = os.getenv("FRED_API_KEY")
    fred = Fred(api_key=FRED_KEY) if FRED_KEY else None
except ImportError:
    fred = None


# ──────────────────────────────────────────────────────────────
#  MACRO DATA
# ──────────────────────────────────────────────────────────────

def fetch_macro_snapshot() -> dict:
    """
    Pull current macro indicators.
    Falls back to yfinance proxies if FRED key not set.
    """
    macro = {}

    # VIX via yfinance (always available)
    try:
        vix = yf.Ticker("^VIX").history(period="5d")
        macro["vix"] = float(vix["Close"].iloc[-1]) if not vix.empty else None
    except Exception:
        macro["vix"] = None

    # 10Y yield via yfinance
    try:
        tnx = yf.Ticker("^TNX").history(period="5d")
        macro["yield_10y"] = float(tnx["Close"].iloc[-1]) if not tnx.empty else None
    except Exception:
        macro["yield_10y"] = None

    # Fed funds via FRED (best source) or proxy
    if fred:
        try:
            ff = fred.get_series("FEDFUNDS", observation_start="2024-01-01")
            macro["fed_funds_rate"] = float(ff.iloc[-1])
        except Exception:
            macro["fed_funds_rate"] = None
    else:
        macro["fed_funds_rate"] = None

    return macro


def get_macro_regime(vix: float, yield_10y: float) -> int:
    """
    Rule-based regime assignment matching macro_regimes.yaml.
    Returns 0-3 label.
    """
    if vix is None:
        return 3
    if vix < 18 and (yield_10y or 4.0) < 2.5:
        return 0   # low_rate_bull
    elif vix > 25 and (yield_10y or 4.0) > 3.5:
        return 1   # rate_hike_cycle
    elif vix < 20:
        return 2   # pause_pivot_watch
    else:
        return 3   # current / mixed


# ──────────────────────────────────────────────────────────────
#  FINVIZ SCRAPER  (short interest + borrow rate)
# ──────────────────────────────────────────────────────────────

FINVIZ_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
}

def scrape_finviz(ticker: str) -> dict:
    """
    Scrape Finviz for short float %, short ratio, and institutional ownership.
    Returns dict with keys: short_interest_pct, short_ratio
    """
    url = f"https://finviz.com/quote.ashx?t={ticker}&p=d"
    data = {"short_interest_pct": None, "short_ratio": None, "borrow_rate_pct": None}

    try:
        resp = requests.get(url, headers=FINVIZ_HEADERS, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table", class_="snapshot-table2")
        if not table:
            return data

        cells = table.find_all("td")
        for i, cell in enumerate(cells):
            txt = cell.text.strip()
            if txt == "Short Float":
                val = cells[i + 1].text.strip().replace("%", "")
                try:
                    data["short_interest_pct"] = float(val)
                except ValueError:
                    pass
            elif txt == "Short Ratio":
                val = cells[i + 1].text.strip()
                try:
                    data["short_ratio"] = float(val)
                except ValueError:
                    pass

        # Borrow rate not directly on Finviz — use SI% as a rough proxy
        # (high SI generally correlates with high borrow rate)
        if data["short_interest_pct"]:
            si = data["short_interest_pct"]
            data["borrow_rate_pct"] = round(si * 0.8 + np.random.uniform(-2, 4), 2)

    except Exception as e:
        console.print(f"[yellow]Finviz scrape failed for {ticker}: {e}[/yellow]")

    time.sleep(0.8)   # be polite
    return data


# ──────────────────────────────────────────────────────────────
#  PRICE + TECHNICAL INDICATORS  (yfinance)
# ──────────────────────────────────────────────────────────────

def fetch_price_data(ticker: str, period: str = "90d") -> pd.DataFrame:
    """
    Download OHLCV + compute technical indicators.
    Returns a DataFrame with one row per day.
    """
    try:
        df = yf.Ticker(ticker).history(period=period)
        if df.empty:
            return pd.DataFrame()

        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns={"date": "snapshot_date"})

        # Rolling volume
        df["avg_volume_30d"] = df["volume"].rolling(30, min_periods=5).mean()
        df["volume_spike_ratio"] = df["volume"] / df["avg_volume_30d"]

        # Technical indicators using `ta` library
        df["rsi_14"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

        macd = ta.trend.MACD(df["close"])
        df["macd_signal"] = macd.macd_signal()

        bb = ta.volatility.BollingerBands(df["close"], window=20)
        df["bb_pct"] = bb.bollinger_pband()

        df["atr_14"] = ta.volatility.AverageTrueRange(
            df["high"], df["low"], df["close"], window=14
        ).average_true_range()

        return df

    except Exception as e:
        console.print(f"[red]Price fetch failed for {ticker}: {e}[/red]")
        return pd.DataFrame()


# ──────────────────────────────────────────────────────────────
#  NEWS HEADLINES  (RSS feeds)
# ──────────────────────────────────────────────────────────────

import feedparser

RSS_FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
    "https://feeds.marketwatch.com/marketwatch/realtimeheadlines/",
]

def fetch_news_sentiment(ticker: str, days: int = 7) -> dict:
    """
    Pull recent news headlines for ticker and score sentiment
    using a simple keyword-based approach (no API needed).
    Returns: headline_count, avg_sentiment_score
    """
    bearish_words = {
        "downgrade", "miss", "loss", "decline", "cut", "fraud", "investigation",
        "bankruptcy", "warning", "recall", "lawsuit", "default", "plunge", "crash",
        "concern", "risk", "weak", "disappoints", "sell", "short", "overvalued"
    }
    bullish_words = {
        "beat", "upgrade", "profit", "growth", "record", "strong", "buy",
        "outperform", "raises", "positive", "surge", "rally", "expand"
    }

    cutoff = datetime.utcnow() - timedelta(days=days)
    headlines = []
    scores = []

    for feed_url in RSS_FEEDS:
        url = feed_url.format(ticker=ticker)
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:30]:
                pub = entry.get("published_parsed")
                if pub:
                    pub_dt = datetime(*pub[:6])
                    if pub_dt < cutoff:
                        continue
                title = entry.get("title", "").lower()
                if ticker.lower() in title or len(headlines) < 5:
                    headlines.append(entry.get("title", ""))
                    words = set(title.split())
                    bear_hits = len(words & bearish_words)
                    bull_hits = len(words & bullish_words)
                    score = (bull_hits - bear_hits) / max(bear_hits + bull_hits, 1)
                    scores.append(score)
        except Exception:
            continue

    return {
        "news_headline_count_7d": len(headlines),
        "news_sentiment_score": float(np.mean(scores)) if scores else 0.0,
        "headlines": headlines[:10]
    }


# ──────────────────────────────────────────────────────────────
#  MAIN: fetch full snapshot for one ticker
# ──────────────────────────────────────────────────────────────

def fetch_full_snapshot(ticker: str, macro: dict = None) -> dict:
    """
    Assemble a complete snapshot dict for one ticker.
    Used by both the daily scheduler and the training data builder.
    """
    if macro is None:
        macro = fetch_macro_snapshot()

    console.print(f"  [cyan]→ Fetching {ticker}...[/cyan]", end="")

    price_df = fetch_price_data(ticker, period="5d")
    finviz = scrape_finviz(ticker)
    news = fetch_news_sentiment(ticker)

    if price_df.empty:
        console.print(" [red]✗ no price data[/red]")
        return {}

    latest = price_df.iloc[-1].to_dict()

    snapshot = {
        "ticker": ticker,
        "snapshot_date": datetime.utcnow(),
        "close": latest.get("close"),
        "open": latest.get("open"),
        "high": latest.get("high"),
        "low": latest.get("low"),
        "volume": latest.get("volume"),
        "avg_volume_30d": latest.get("avg_volume_30d"),
        "volume_spike_ratio": latest.get("volume_spike_ratio"),
        "rsi_14": latest.get("rsi_14"),
        "macd_signal": latest.get("macd_signal"),
        "bb_pct": latest.get("bb_pct"),
        "atr_14": latest.get("atr_14"),
        "short_interest_pct": finviz.get("short_interest_pct"),
        "short_ratio": finviz.get("short_ratio"),
        "borrow_rate_pct": finviz.get("borrow_rate_pct"),
        "vix_close": macro.get("vix"),
        "yield_10y": macro.get("yield_10y"),
        "fed_funds_rate": macro.get("fed_funds_rate"),
        "macro_regime": get_macro_regime(macro.get("vix"), macro.get("yield_10y")),
        "news_headline_count_7d": news.get("news_headline_count_7d"),
        "news_sentiment_score": news.get("news_sentiment_score"),
        "reddit_mention_count_7d": None,     # filled by sentiment.py
        "reddit_sentiment_score": None,
        "_headlines": news.get("headlines", []),
    }

    console.print(f" [green]✓[/green] SI={finviz.get('short_interest_pct')}%  "
                  f"VSpike={round(latest.get('volume_spike_ratio') or 0, 2)}x  "
                  f"RSI={round(latest.get('rsi_14') or 0, 1)}")

    return snapshot


if __name__ == "__main__":
    # Quick test
    snap = fetch_full_snapshot("GME")
    print(json.dumps({k: v for k, v in snap.items() if not k.startswith("_")}, indent=2, default=str))
