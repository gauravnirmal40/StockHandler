"""
build_training_data.py — Fetch historical data and label squeeze setups

Run this ONCE before training to populate the database with labeled
historical snapshots. Takes ~20-40 minutes depending on watchlist size.

What it does:
  1. For each ticker, downloads 2 years of daily OHLCV
  2. Computes all technical indicators per day
  3. Attaches macro regime label from macro_regimes.yaml
  4. Labels each day as squeeze_occurred=True/False based on
     subsequent price movement and short interest change
  5. Saves everything to ticker_snapshots table

Usage:
  python build_training_data.py
  python build_training_data.py --ticker GME     # single ticker
  python build_training_data.py --period 3y      # 3 years of history
"""

import argparse
import json
import os
import yaml
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
import ta

from utils.db import init_db, get_session, TickerSnapshot
from utils.fetcher import scrape_finviz, fetch_news_sentiment, get_macro_regime

load_dotenv()
console = Console()


def load_regimes(path="data/macro_regimes.yaml") -> list:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg["regimes"], cfg["squeeze_label_thresholds"]


def assign_regime(date: datetime, regimes: list) -> int:
    for r in regimes:
        start = datetime.fromisoformat(r["start"])
        end = datetime.now() if r["end"] is None else datetime.fromisoformat(r["end"])
        if start <= date <= end:
            return r["label"]
    return 3


def label_squeeze(df: pd.DataFrame, thresholds: dict, window: int = 10) -> pd.DataFrame:
    """
    Look forward `window` days from each row and label squeeze_occurred.
    A squeeze is labeled True if:
      - future max price change > price_move_pct threshold, OR
      - volume spike ratio > threshold in next window days
    """
    df = df.copy().reset_index(drop=True)
    labels = []

    for i in range(len(df)):
        future = df.iloc[i + 1 : i + 1 + window]
        if future.empty:
            labels.append(None)
            continue

        current_close = df.iloc[i]["close"]
        max_future = future["close"].max()
        pct_change = ((max_future - current_close) / current_close) * 100

        volume_spike = future["volume_spike_ratio"].max() if "volume_spike_ratio" in future else 0

        squeeze = (
            pct_change >= thresholds["price_move_pct"]
            or volume_spike >= thresholds["volume_spike_ratio"]
        )
        labels.append(bool(squeeze))

    df["squeeze_occurred"] = labels
    df["price_change_10d"] = (
        df["close"].shift(-window) - df["close"]
    ) / df["close"] * 100

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicator columns."""
    df = df.copy()
    df["avg_volume_30d"] = df["volume"].rolling(30, min_periods=5).mean()
    df["volume_spike_ratio"] = df["volume"] / df["avg_volume_30d"]

    df["rsi_14"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

    macd = ta.trend.MACD(df["close"])
    df["macd_signal"] = macd.macd_signal()

    bb = ta.volatility.BollingerBands(df["close"], window=20)
    df["bb_pct"] = bb.bollinger_pband()

    df["atr_14"] = ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"], window=14
    ).average_true_range()

    return df


def process_ticker(ticker: str, period: str, regimes: list, thresholds: dict,
                   session, si_data: dict = None) -> int:
    """
    Download, enrich, label, and save historical data for one ticker.
    Returns number of rows inserted.
    """
    console.print(f"\n[bold cyan]Processing {ticker}[/bold cyan]")

    # Download historical OHLCV
    hist = yf.Ticker(ticker).history(period=period)
    if hist.empty:
        console.print(f"  [red]No data for {ticker}[/red]")
        return 0

    hist = hist.reset_index()
    hist.columns = [c.lower() for c in hist.columns]
    hist = hist.rename(columns={"date": "snapshot_date"})

    # Compute features
    hist = build_features(hist)

    # Label squeezes
    hist = label_squeeze(hist, thresholds)

    # Get macro proxies via yfinance (VIX, 10Y)
    vix_hist = yf.Ticker("^VIX").history(period=period)[["Close"]].rename(columns={"Close": "vix"})
    tnx_hist = yf.Ticker("^TNX").history(period=period)[["Close"]].rename(columns={"Close": "yield_10y"})

    # Finviz — only current snapshot (historical SI not free)
    current_si = si_data or scrape_finviz(ticker)

    inserted = 0
    for _, row in hist.iterrows():
        date = pd.Timestamp(row["snapshot_date"]).to_pydatetime().replace(tzinfo=None)

        # VIX and yield at that date (nearest available)
        vix_val = None
        yield_val = None
        try:
            close_dates = vix_hist.index
            if hasattr(close_dates[0], "tz_localize"):
                vix_val = float(vix_hist.iloc[vix_hist.index.get_indexer([date], method="nearest")[0]]["vix"])
        except Exception:
            pass
        try:
            yield_val = float(tnx_hist.iloc[tnx_hist.index.get_indexer([date], method="nearest")[0]]["yield_10y"])
        except Exception:
            pass

        regime = assign_regime(date, regimes)

        snap = TickerSnapshot(
            ticker=ticker,
            snapshot_date=date,
            close=_safe(row.get("close")),
            open=_safe(row.get("open")),
            high=_safe(row.get("high")),
            low=_safe(row.get("low")),
            volume=_safe(row.get("volume")),
            avg_volume_30d=_safe(row.get("avg_volume_30d")),
            volume_spike_ratio=_safe(row.get("volume_spike_ratio")),
            rsi_14=_safe(row.get("rsi_14")),
            macd_signal=_safe(row.get("macd_signal")),
            bb_pct=_safe(row.get("bb_pct")),
            atr_14=_safe(row.get("atr_14")),
            short_interest_pct=current_si.get("short_interest_pct"),
            short_ratio=current_si.get("short_ratio"),
            borrow_rate_pct=current_si.get("borrow_rate_pct"),
            vix_close=vix_val,
            yield_10y=yield_val,
            fed_funds_rate=None,
            macro_regime=regime,
            reddit_mention_count_7d=None,
            reddit_sentiment_score=None,
            news_headline_count_7d=None,
            news_sentiment_score=None,
            squeeze_occurred=row.get("squeeze_occurred"),
            price_change_10d=_safe(row.get("price_change_10d")),
        )
        session.add(snap)
        inserted += 1

    session.commit()
    console.print(f"  [green]✓ Inserted {inserted} rows[/green]")
    return inserted


def _safe(val):
    if val is None:
        return None
    try:
        if np.isnan(float(val)):
            return None
        return float(val)
    except (TypeError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser(description="Build training data")
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--period", type=str, default="2y",
                        help="yfinance period: 1y, 2y, 3y, 5y")
    parser.add_argument("--watchlist", type=str, default="data/watchlist.csv")
    args = parser.parse_args()

    engine = init_db()
    session = get_session(engine)
    regimes, thresholds = load_regimes()

    if args.ticker:
        tickers = [args.ticker.upper()]
    else:
        wl = pd.read_csv(args.watchlist)
        tickers = wl["ticker"].tolist()

    console.print(f"\n[bold]Building training data for {len(tickers)} tickers "
                  f"({args.period} history)[/bold]\n")

    total_rows = 0
    for ticker in tickers:
        rows = process_ticker(ticker, args.period, regimes, thresholds, session)
        total_rows += rows

    console.print(f"\n[bold green]Done! Total rows inserted: {total_rows}[/bold green]")


if __name__ == "__main__":
    main()
