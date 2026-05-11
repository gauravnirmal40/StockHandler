# Short Squeeze Scanner
# 📌 Squeeze Risk Signal Engine

## Overview
Squeeze Risk Signal Engine is a lightweight decision system that identifies potential short squeeze setups by converting fragmented market signals into a single ranked risk output.

It is designed as a **fast MVP tool for evaluating asymmetric market risk**, not a trading bot or production financial system.

---

## 🎯 Problem
Short squeeze opportunities are difficult to detect early because key signals are scattered across:
- price and volume movement
- short interest data
- social sentiment noise

Individually, these signals are weak. Together, they can indicate early-stage squeeze conditions—but are difficult to interpret in real time.

---

## 💡 Solution
This system aggregates multiple weak signals into a **single interpretable risk score per stock**, enabling fast identification of potential squeeze setups.

Instead of analyzing raw data, users get a **ranked list of opportunities with explanations**.

---

## 🔄 System Workflow

Watchlist Input  
↓  
Market Data Collection (price, volume, short interest)  
↓  
Sentiment Signals (Reddit + news context)  
↓  
Feature Extraction (signal indicators)  
↓  
Risk Scoring Model (ML-based classification)  
↓  
Signal Aggregation Engine  
↓  
Ranked Squeeze Risk Output  

---

## 📊 Output Example

The system generates a ranked view of stocks based on squeeze probability.

### Live Dashboard Output

![Squeeze Risk Dashboard](assets/dashboard1.png)
![Squeeze Risk Dashboard](assets/dashboard2.png)

### Interpretation

- **GME → High Risk**  
  Strong borrow rate + volume spike + elevated retail sentiment

- **AMC → Medium Risk**  
  Mixed signals with weak confirmation from price action

- **AAPL → Low Risk**  
  Stable fundamentals, no squeeze indicators detected

---

## 🧠 What the System Produces

Each scan outputs:
- Ranked list of stocks by squeeze risk
- Risk level classification (High / Medium / Low)
- Explanation of contributing signals per stock

The goal is not raw prediction accuracy, but **fast interpretability under uncertainty**.

---

## 🧱 Design Principles

- Signal compression over data complexity  
- Interpretability over black-box prediction  
- Fast MVP iteration over production robustness  
- Single clear output over multiple dashboards  

---

## ⚙️ Tech Stack

- Python  
- Pandas / NumPy  
- XGBoost (risk classification model)  
- Streamlit (dashboard UI)  
- Reddit API + news scraping  
- Yahoo Finance / market data sources  

---

## 🚀 How to Run

```bash
pip install -r requirements.txt

# Run full pipeline
python run_scanner.py

# Launch dashboard
streamlit run dashboard.py

# Scan a single ticker
python run_scanner.py --ticker GME

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
