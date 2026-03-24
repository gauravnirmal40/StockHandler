"""
bear_case_writer.py — Generate AI bear case summaries using Groq (free)

Takes a snapshot dict + squeeze probability and writes a crisp,
analytical bear case for the ticker.  Output is stored in
prediction_results.bear_case_summary.

Uses Groq's free API with LLaMA 3.1 70B — fast, free, no credit card.
Get your key at: console.groq.com  (takes 2 minutes)

The prompt is deliberately styled to sound like a personal
trading journal entry, not a formal report — because this is
a personal project.
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from rich.console import Console
from groq import Groq

load_dotenv()
console = Console()

# Model options (all free on Groq):
#   "llama-3.1-70b-versatile"   ← best quality, use this
#   "llama-3.1-8b-instant"      ← faster, slightly worse
#   "mixtral-8x7b-32768"        ← good alternative
GROQ_MODEL = "llama-3.1-70b-versatile"

_client = None

def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not set.\n"
                "  1. Go to console.groq.com and sign up (free)\n"
                "  2. Create an API key\n"
                "  3. Add GROQ_API_KEY=your_key to your .env file"
            )
        _client = Groq(api_key=api_key)
    return _client


BEAR_CASE_PROMPT = """You are writing a personal trading journal entry for a short-seller analyzing {ticker}.

Here is the current signal snapshot:
- Squeeze probability score: {squeeze_prob:.0%}
- Short interest: {si_pct}% of float
- Short ratio (days to cover): {short_ratio}
- Borrow rate: {borrow_rate}%
- Volume spike ratio: {vol_spike}x (vs 30-day avg)
- RSI-14: {rsi}
- Bollinger band %: {bb_pct}
- VIX: {vix}
- 10Y yield: {yield_10y}%
- Macro regime: {macro_regime}
- News sentiment (7d): {news_sentiment} ({news_count} headlines)
- Reddit sentiment (7d): {reddit_sentiment} ({reddit_mentions} mentions)

Recent news headlines:
{headlines}

Write a SHORT bear case analysis (3-4 paragraphs max) in a personal, analytical style — like a trader writing in their own notebook. Not formal. Not hedged with boilerplate disclaimers. Use specific numbers from the data above. Focus on:
1. The core structural bear thesis for this stock
2. What the short interest and borrow rate signal about conviction
3. What macro conditions mean for this specific setup
4. One key risk / what would invalidate the thesis

Sign off with your personal conviction level: STRONG BEAR / MODERATE BEAR / SPECULATIVE / NEUTRAL.
Do not include any disclaimers, legal language, or "this is not financial advice" text.
Keep it tight — a trader reading this in 90 seconds should know exactly what to think.
"""


def _fmt(val, decimals=1, fallback="N/A"):
    if val is None:
        return fallback
    try:
        return round(float(val), decimals)
    except (TypeError, ValueError):
        return fallback


def write_bear_case(snapshot: dict, squeeze_prob: float) -> str:
    """
    Generate a bear case writeup for a ticker using Claude.

    Args:
        snapshot: dict from fetcher.fetch_full_snapshot()
        squeeze_prob: float 0.0-1.0 from model

    Returns:
        str: the bear case text
    """
    ticker = snapshot.get("ticker", "UNKNOWN")
    headlines = snapshot.get("_headlines", [])
    headlines_str = "\n".join(f"- {h}" for h in headlines[:6]) if headlines else "- No headlines in the last 7 days"

    macro_labels = {
        0: "Low-rate bull / retail mania era",
        1: "Rate hike cycle / multiple compression",
        2: "Pause / pivot watch",
        3: "Current / mixed",
    }
    macro_label = macro_labels.get(snapshot.get("macro_regime"), "Unknown")

    prompt = BEAR_CASE_PROMPT.format(
        ticker=ticker,
        squeeze_prob=squeeze_prob,
        si_pct=_fmt(snapshot.get("short_interest_pct")),
        short_ratio=_fmt(snapshot.get("short_ratio")),
        borrow_rate=_fmt(snapshot.get("borrow_rate_pct")),
        vol_spike=_fmt(snapshot.get("volume_spike_ratio")),
        rsi=_fmt(snapshot.get("rsi_14")),
        bb_pct=_fmt(snapshot.get("bb_pct")),
        vix=_fmt(snapshot.get("vix_close")),
        yield_10y=_fmt(snapshot.get("yield_10y")),
        macro_regime=macro_label,
        news_sentiment=_fmt(snapshot.get("news_sentiment_score"), decimals=2),
        news_count=_fmt(snapshot.get("news_headline_count_7d"), decimals=0),
        reddit_sentiment=_fmt(snapshot.get("reddit_sentiment_score"), decimals=2),
        reddit_mentions=_fmt(snapshot.get("reddit_mention_count_7d"), decimals=0),
        headlines=headlines_str,
    )

    console.print(f"  [dim]Generating bear case for {ticker}...[/dim]", end="")

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=600,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content.strip()
        console.print(" [green]✓[/green]")
        return text

    except Exception as e:
        console.print(f" [red]✗ {e}[/red]")
        return f"Bear case generation failed: {e}"


def batch_write_bear_cases(results: list) -> list:
    """
    results: list of dicts, each with keys:
        snapshot, squeeze_prob
    Adds 'bear_case' key to each dict in place.
    """
    for r in results:
        r["bear_case"] = write_bear_case(r["snapshot"], r["squeeze_prob"])
    return results


if __name__ == "__main__":
    # Quick test with fake data
    fake_snapshot = {
        "ticker": "GME",
        "short_interest_pct": 22.5,
        "short_ratio": 3.2,
        "borrow_rate_pct": 18.4,
        "volume_spike_ratio": 2.8,
        "rsi_14": 68.2,
        "bb_pct": 0.82,
        "vix_close": 19.4,
        "yield_10y": 4.35,
        "macro_regime": 3,
        "news_sentiment_score": -0.45,
        "news_headline_count_7d": 8,
        "reddit_sentiment_score": 0.62,
        "reddit_mention_count_7d": 234,
        "_headlines": [
            "GameStop misses Q3 earnings estimates",
            "GME short sellers add positions amid rally",
            "Roaring Kitty returns to social media",
        ],
    }
    result = write_bear_case(fake_snapshot, 0.71)
    print("\n" + result)
