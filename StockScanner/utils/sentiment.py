"""
sentiment.py — Reddit mention tracking and sentiment scoring
Uses PRAW (Python Reddit API Wrapper).

Subreddits monitored: wallstreetbets, stocks, investing, shortsqueeze, stockmarket
"""

import os
import re
import time
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
console = Console()

# Positive/negative word lists tuned for financial Reddit
BULLISH_SLANG = {
    "moon", "mooning", "squeeze", "short squeeze", "gamma squeeze", "yolo",
    "rockets", "tendies", "calls", "buy", "long", "bullish", "undervalued",
    "breakout", "catalyst", "covering", "apes", "hodl", "diamond hands"
}
BEARISH_SLANG = {
    "puts", "short", "puts printing", "bankrupt", "bagholders", "dead cat",
    "overvalued", "dilution", "fraud", "sec", "investigation", "delisted",
    "reverse split", "bag", "drill", "crater", "dump", "cash burn", "bankruptcy"
}

SUBREDDITS = [
    "wallstreetbets",
    "stocks",
    "investing",
    "shortsqueeze",
    "stockmarket",
    "options",
    "pennystocks",
]


def _get_reddit_client():
    """Initialize PRAW client from env vars."""
    try:
        import praw
        return praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID", ""),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET", ""),
            user_agent=os.getenv("REDDIT_USER_AGENT", "squeeze_scanner/1.0"),
        )
    except Exception as e:
        console.print(f"[yellow]Reddit client init failed: {e}[/yellow]")
        return None


def score_text(text: str) -> float:
    """
    Simple lexicon-based sentiment score.
    Returns value in [-1.0, 1.0].
    Negative = bearish, Positive = bullish.
    """
    text_lower = text.lower()
    words = set(re.findall(r"\b\w+\b", text_lower))

    bull_hits = len(words & BULLISH_SLANG)
    bear_hits = len(words & BEARISH_SLANG)

    total = bull_hits + bear_hits
    if total == 0:
        return 0.0

    return round((bull_hits - bear_hits) / total, 3)


def fetch_reddit_sentiment(ticker: str, days: int = 7) -> dict:
    """
    Search Reddit for mentions of ticker over last N days.
    Returns dict: mention_count, avg_sentiment, sample_posts
    """
    reddit = _get_reddit_client()

    if not reddit:
        # Return neutral defaults if Reddit not configured
        return {
            "reddit_mention_count_7d": 0,
            "reddit_sentiment_score": 0.0,
            "sample_posts": [],
        }

    cutoff_ts = time.time() - (days * 86400)
    mentions = []
    scores = []
    sample_posts = []

    search_terms = [
        f"${ticker}",
        ticker,
        f"{ticker} short",
        f"{ticker} puts",
    ]

    try:
        for subreddit_name in SUBREDDITS:
            subreddit = reddit.subreddit(subreddit_name)

            for term in search_terms[:2]:   # limit to avoid rate limits
                try:
                    results = subreddit.search(
                        term,
                        sort="new",
                        time_filter="week",
                        limit=25,
                    )

                    for post in results:
                        if post.created_utc < cutoff_ts:
                            continue
                        if ticker.upper() not in (post.title + " " + post.selftext).upper():
                            continue

                        text = post.title + " " + post.selftext
                        score = score_text(text)
                        mentions.append(post.id)
                        scores.append(score)

                        if len(sample_posts) < 5:
                            sample_posts.append({
                                "title": post.title[:120],
                                "score": post.score,
                                "sentiment": score,
                                "url": f"https://reddit.com{post.permalink}",
                                "subreddit": subreddit_name,
                            })

                    time.sleep(0.3)

                except Exception:
                    continue

    except Exception as e:
        console.print(f"[yellow]Reddit search error for {ticker}: {e}[/yellow]")

    # Deduplicate by post ID
    unique_count = len(set(mentions))

    return {
        "reddit_mention_count_7d": unique_count,
        "reddit_sentiment_score": float(np.mean(scores)) if scores else 0.0,
        "sample_posts": sample_posts,
    }


def enrich_snapshot_with_reddit(snapshot: dict) -> dict:
    """Add Reddit sentiment fields to an existing snapshot dict."""
    ticker = snapshot.get("ticker", "")
    if not ticker:
        return snapshot

    reddit_data = fetch_reddit_sentiment(ticker)
    snapshot["reddit_mention_count_7d"] = reddit_data["reddit_mention_count_7d"]
    snapshot["reddit_sentiment_score"] = reddit_data["reddit_sentiment_score"]
    snapshot["_sample_posts"] = reddit_data["sample_posts"]
    return snapshot


if __name__ == "__main__":
    import json
    result = fetch_reddit_sentiment("GME", days=7)
    print(json.dumps(result, indent=2))
