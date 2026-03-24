"""
run_scanner.py — Daily scan orchestrator

Runs the full pipeline for all tickers on the watchlist:
  1. Fetch latest market data + Finviz short data
  2. Pull Reddit sentiment
  3. Score each ticker through the ML model
  4. Generate bear case writeups via Claude
  5. Save prediction_results to DB
  6. Trigger alerts for high-conviction signals

Usage:
  python run_scanner.py                     # scan all watchlist tickers
  python run_scanner.py --ticker GME        # single ticker
  python run_scanner.py --no-ai             # skip Claude bear cases
  python run_scanner.py --alert-threshold 0.7
"""

import argparse
import json
import os
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from utils.db import init_db, get_session, PredictionResult
from utils.fetcher import fetch_full_snapshot, fetch_macro_snapshot
from utils.sentiment import enrich_snapshot_with_reddit
from train_model import predict_proba, FEATURE_COLS
from bear_case_writer import write_bear_case

load_dotenv()
console = Console()

ALERT_THRESHOLD = float(os.getenv("ALERT_THRESHOLD", "0.65"))
WATCHLIST_PATH = "data/watchlist.csv"


# ──────────────────────────────────────────────────────────────
#  EMAIL ALERT
# ──────────────────────────────────────────────────────────────

def send_email_alert(results: list):
    """Send email summary of high-conviction signals."""
    email = os.getenv("ALERT_EMAIL")
    password = os.getenv("ALERT_EMAIL_PASSWORD")
    smtp_host = os.getenv("ALERT_SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("ALERT_SMTP_PORT", 587))

    if not email or not password:
        console.print("[yellow]Email alerts not configured. Set ALERT_EMAIL and ALERT_EMAIL_PASSWORD in .env[/yellow]")
        return

    high_conviction = [r for r in results if r["squeeze_prob"] >= ALERT_THRESHOLD]
    if not high_conviction:
        return

    subject = f"Squeeze Scanner Alert — {len(high_conviction)} signal(s) [{datetime.now().strftime('%b %d')}]"

    body_lines = [f"High-conviction squeeze setups — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"]
    for r in sorted(high_conviction, key=lambda x: x["squeeze_prob"], reverse=True):
        body_lines.append(
            f"{'═'*50}\n"
            f"  {r['ticker']}  —  Score: {r['squeeze_prob']:.0%}\n"
            f"  SI: {r['snapshot'].get('short_interest_pct')}%  |  "
            f"Vol spike: {round(r['snapshot'].get('volume_spike_ratio') or 0, 1)}x  |  "
            f"Borrow: {r['snapshot'].get('borrow_rate_pct')}%\n\n"
            f"{r.get('bear_case', 'N/A')}\n"
        )

    msg = MIMEMultipart()
    msg["From"] = email
    msg["To"] = email
    msg["Subject"] = subject
    msg.attach(MIMEText("\n".join(body_lines), "plain"))

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.login(email, password)
            server.send_message(msg)
        console.print(f"[green]Alert email sent to {email}[/green]")
    except Exception as e:
        console.print(f"[red]Email failed: {e}[/red]")


# ──────────────────────────────────────────────────────────────
#  MAIN SCAN
# ──────────────────────────────────────────────────────────────

def run_scan(tickers: list, use_ai: bool = True, alert_threshold: float = None):
    threshold = alert_threshold or ALERT_THRESHOLD
    session = get_session()
    run_date = datetime.utcnow()

    # Fetch macro once for all tickers
    console.print("\n[bold]Fetching macro snapshot...[/bold]")
    macro = fetch_macro_snapshot()
    console.print(
        f"  VIX: {macro.get('vix')}  |  10Y: {macro.get('yield_10y')}%  |  "
        f"Fed funds: {macro.get('fed_funds_rate')}%"
    )

    console.print(f"\n[bold]Scanning {len(tickers)} tickers...[/bold]")

    results = []
    for ticker in tickers:
        try:
            # 1. Market data
            snap = fetch_full_snapshot(ticker, macro=macro)
            if not snap:
                continue

            # 2. Reddit sentiment
            snap = enrich_snapshot_with_reddit(snap)

            # 3. ML score
            try:
                prob = predict_proba(snap)
            except FileNotFoundError:
                console.print("  [yellow]No trained model found. Run train_model.py first.[/yellow]")
                prob = 0.0

            # 4. Bear case (optional)
            bear_case = ""
            if use_ai and prob >= 0.40:   # only generate for semi-interesting signals
                bear_case = write_bear_case(snap, prob)

            results.append({
                "ticker": ticker,
                "squeeze_prob": prob,
                "snapshot": snap,
                "bear_case": bear_case,
                "alert_triggered": prob >= threshold,
            })

            # 5. Save to DB
            pr = PredictionResult(
                ticker=ticker,
                run_date=run_date,
                squeeze_probability=prob,
                bear_case_summary=bear_case,
                alert_triggered=prob >= threshold,
                features_json=json.dumps(
                    {k: snap.get(k) for k in FEATURE_COLS}, default=str
                ),
                model_version="latest",
            )
            session.add(pr)

        except Exception as e:
            console.print(f"  [red]Error on {ticker}: {e}[/red]")
            continue

    session.commit()

    # ── Rank and display results ──────────────────────────────
    results.sort(key=lambda x: x["squeeze_prob"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    print_results_table(results, threshold)

    # ── Alerts ───────────────────────────────────────────────
    triggered = [r for r in results if r["alert_triggered"]]
    if triggered:
        console.print(f"\n[bold red]⚡ {len(triggered)} alert(s) triggered[/bold red]")
        send_email_alert(triggered)

    return results


def print_results_table(results: list, threshold: float):
    table = Table(
        title=f"Squeeze Scanner — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Rank", style="dim", width=5)
    table.add_column("Ticker", style="bold", width=8)
    table.add_column("Score", width=8)
    table.add_column("SI %", width=7)
    table.add_column("Vol Spike", width=9)
    table.add_column("Borrow %", width=9)
    table.add_column("RSI", width=6)
    table.add_column("Reddit", width=8)
    table.add_column("Alert", width=7)

    for r in results:
        s = r["snapshot"]
        prob = r["squeeze_prob"]
        score_str = f"{prob:.0%}"
        alert_str = "🔴 YES" if r["alert_triggered"] else "—"

        style = ""
        if prob >= threshold:
            style = "bold red"
        elif prob >= 0.50:
            style = "yellow"

        table.add_row(
            str(r.get("rank", "?")),
            r["ticker"],
            score_str,
            str(s.get("short_interest_pct") or "—"),
            f"{round(s.get('volume_spike_ratio') or 0, 1)}x",
            str(s.get("borrow_rate_pct") or "—"),
            str(round(s.get("rsi_14") or 0, 1)),
            str(s.get("reddit_mention_count_7d") or "—"),
            alert_str,
            style=style,
        )

    console.print("\n")
    console.print(table)


def load_watchlist(path: str) -> list:
    df = pd.read_csv(path)
    return df["ticker"].tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run squeeze scanner")
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--no-ai", action="store_true", help="Skip Claude bear cases")
    parser.add_argument("--alert-threshold", type=float, default=None)
    args = parser.parse_args()

    init_db()

    if args.ticker:
        tickers = [args.ticker.upper()]
    else:
        tickers = load_watchlist(WATCHLIST_PATH)

    run_scan(
        tickers=tickers,
        use_ai=not args.no_ai,
        alert_threshold=args.alert_threshold,
    )
