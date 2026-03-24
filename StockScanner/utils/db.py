"""
db.py — Database setup for squeeze scanner
SQLite via SQLAlchemy. Run this once to initialize.
"""

import os
from sqlalchemy import (
    create_engine, Column, Integer, Float, String,
    DateTime, Boolean, Text, ForeignKey, Index
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
Base = declarative_base()


class TickerSnapshot(Base):
    """Daily market snapshot per ticker — core feature table"""
    __tablename__ = "ticker_snapshots"

    id = Column(Integer, primary_key=True)
    ticker = Column(String(10), nullable=False)
    snapshot_date = Column(DateTime, nullable=False)

    # Price & volume
    close = Column(Float)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    volume = Column(Float)
    avg_volume_30d = Column(Float)
    volume_spike_ratio = Column(Float)      # volume / avg_volume_30d

    # Short data (from Finviz / yfinance)
    short_interest_pct = Column(Float)      # % of float sold short
    short_ratio = Column(Float)             # days to cover
    borrow_rate_pct = Column(Float)         # annualized cost to borrow

    # Technical indicators
    rsi_14 = Column(Float)
    macd_signal = Column(Float)
    bb_pct = Column(Float)                  # Bollinger band % position
    atr_14 = Column(Float)                  # Average True Range

    # Macro regime label
    macro_regime = Column(Integer)          # 0-3 from macro_regimes.yaml
    vix_close = Column(Float)
    yield_10y = Column(Float)
    fed_funds_rate = Column(Float)

    # Sentiment
    reddit_mention_count_7d = Column(Integer)
    reddit_sentiment_score = Column(Float)  # -1 to 1
    news_headline_count_7d = Column(Integer)
    news_sentiment_score = Column(Float)

    # Label (for training only — set after the fact)
    squeeze_occurred = Column(Boolean, nullable=True)   # did squeeze happen?
    price_change_10d = Column(Float, nullable=True)     # actual outcome

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_ticker_date", "ticker", "snapshot_date"),
    )


class PredictionResult(Base):
    """Model output per ticker per run"""
    __tablename__ = "prediction_results"

    id = Column(Integer, primary_key=True)
    ticker = Column(String(10), nullable=False)
    run_date = Column(DateTime, nullable=False)
    squeeze_probability = Column(Float)         # 0.0 - 1.0
    signal_rank = Column(Integer)               # rank across watchlist
    bear_case_summary = Column(Text)            # Claude-generated writeup
    alert_triggered = Column(Boolean, default=False)
    features_json = Column(Text)                # JSON dump of input features
    model_version = Column(String(50))

    created_at = Column(DateTime, default=datetime.utcnow)


class SentimentLog(Base):
    """Raw sentiment hits for auditing"""
    __tablename__ = "sentiment_log"

    id = Column(Integer, primary_key=True)
    ticker = Column(String(10), nullable=False)
    source = Column(String(20))                 # 'reddit' or 'news'
    headline = Column(Text)
    sentiment_score = Column(Float)
    url = Column(String(500))
    published_at = Column(DateTime)
    captured_at = Column(DateTime, default=datetime.utcnow)


class ModelVersion(Base):
    """Track trained model versions"""
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True)
    version_tag = Column(String(50))
    trained_at = Column(DateTime)
    training_samples = Column(Integer)
    test_auc = Column(Float)
    test_precision = Column(Float)
    test_recall = Column(Float)
    feature_importance_json = Column(Text)
    notes = Column(Text)


def init_db():
    db_path = os.getenv("DB_PATH", "data/squeeze_scanner.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Base.metadata.create_all(engine)
    print(f"[db] Database initialized at {db_path}")
    return engine


def get_session(engine=None):
    if engine is None:
        db_path = os.getenv("DB_PATH", "data/squeeze_scanner.db")
        engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Session = sessionmaker(bind=engine)
    return Session()


if __name__ == "__main__":
    init_db()
    print("[db] Tables created successfully.")
