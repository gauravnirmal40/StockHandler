# Short Squeeze Scanner
### Personal watchlist monitor for short-selling setups

Short Squeeze Scanner is a lightweight market signal tool that identifies potential short squeeze setups by combining price action, short interest data, and sentiment signals into a single ranked risk score.

It is designed as a rapid MVP system for detecting asymmetric market risk opportunities, not a trading bot or production financial system.

# Problem

Short squeeze setups are typically identified late because signal sources are fragmented:

short interest data (delayed / incomplete)
social sentiment (noisy)
price/volume movement (reactive)

This makes it hard to quickly evaluate whether a stock is entering a high-risk squeeze environment.
---

## What it does

- Tracks short interest %, borrow rate, and volume spikes via Yahoo Finance + Finviz
- Pulls Reddit (WSB, r/stocks, r/shortsqueeze) and news headline sentiment daily
- Labels historical data by macro regime (rate hike cycle, bull run, etc.)
- Trains an XGBoost classifier on past squeeze setups to predict new ones
- Generates a bear case writeup per ticker using Claude API
- Shows everything on a Streamlit dashboard

---

## What you need (inputs)

### 1. API Keys (add to `.env`)
| Key | Where to get it |
|-----|----------------|------|
| `ANTHROPIC_API_KEY` | console.anthropic.com |
| `REDDIT_CLIENT_ID` + `SECRET` | reddit.com/prefs/apps  |
| `FRED_API_KEY` | fred.stlouisfed.org/docs/api |
| `ALERT_EMAIL` + password | Gmail app password |

### 2. Your watchlist (`data/watchlist.csv`)
Edit this file to add/remove tickers you actually follow. Columns:
- `ticker` — stock symbol
- `sector` — for your own reference
- `why_watching` — personal note
- `short_thesis` — your bear thesis (shows in dashboard)
- `date_added` — when you started tracking it

### 3. Architecture
Market Data (price, short interest)
            ↓
Sentiment Layer (Reddit + news)
            ↓
Feature Engineering (signals extraction)
            ↓
Risk Scoring Model (XGBoost classifier)
            ↓
Signal Aggregation Engine
            ↓
Output Layer (CLI / Streamlit dashboard)

### 4. Macro regimes (`data/macro_regimes.yaml`)
Already configured with 4 regimes from 2020-present.
You can add new ones or adjust the date ranges.

---

## Setup

```bash
# Clone / navigate to project folder
cd squeeze_scanner

# Install dependencies
pip install -r requirements.txt

# Copy and fill in your API keys
cp .env.template .env
# Edit .env with your keys

# Initialize the database
python utils/db.py

# Build historical training data (takes ~20-40 mins)
python build_training_data.py --period 2y

# Train the model
python train_model.py

# Run a scan
python run_scanner.py

# Launch the dashboard
streamlit run dashboard.py
```

---

## Daily usage

```bash
# Quick scan + open dashboard
python run_scanner.py && streamlit run dashboard.py

# Single ticker
python run_scanner.py --ticker GME

# Skip AI bear cases (faster)
python run_scanner.py --no-ai
```
## Tech Stack
Python
Pandas / NumPy
XGBoost (classification model)
Streamlit (dashboard)
Reddit API + news scraping
Yahoo Finance / Finviz data sources

## What Makes This Different

Most tools either:

track sentiment only
or track price only
or over-engineer ML without interpretability

This system focuses on:

combining weak signals into one interpretable risk layer

## Things I still want to add

- [ ] Options flow data (unusual options activity)
- [ ] Automated regime detection instead of manual YAML
- [ ] Backtesting module to check historical signal accuracy
- [ ] SMS alerts (Twilio)
- [ ] More sectors in watchlist (currently heavy consumer/tech)
- [ ] Fix sentiment scoring — it's too naive right now

---

## Known issues

- Finviz scraping sometimes fails if they update their HTML
- Reddit rate limits hit if scanning too many tickers at once
- Historical short interest data not available free — only current SI used for training
- Model trained on limited data (watchlist only) — would be better with broader universe

---

*Personal project. Not financial advice. Built for my own trading research.*
