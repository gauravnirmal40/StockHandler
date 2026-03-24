"""
train_model.py — Train the squeeze prediction model

Features used:
  - Short interest %, short ratio, borrow rate
  - Volume spike ratio
  - RSI, MACD signal, Bollinger band %
  - VIX, 10Y yield, macro regime
  - News sentiment score, Reddit sentiment score

Target: squeeze_occurred (binary classification)

Usage:
  python train_model.py
  python train_model.py --model xgboost        # default
  python train_model.py --model logistic        # simpler, more explainable
  python train_model.py --eval                  # print metrics only, don't save
"""

import argparse
import json
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_score, recall_score
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from utils.db import init_db, get_session, TickerSnapshot, ModelVersion

load_dotenv()
console = Console()

MODEL_PATH = "models/squeeze_model.pkl"
SCALER_PATH = "models/scaler.pkl"

# ── Feature columns used for training ──────────────────────────
FEATURE_COLS = [
    "short_interest_pct",
    "short_ratio",
    "borrow_rate_pct",
    "volume_spike_ratio",
    "rsi_14",
    "macd_signal",
    "bb_pct",
    "atr_14",
    "vix_close",
    "yield_10y",
    "macro_regime",
    "news_sentiment_score",
    "reddit_sentiment_score",
    "news_headline_count_7d",
    "reddit_mention_count_7d",
]


def load_training_data(session) -> pd.DataFrame:
    """Pull labeled rows from DB into a DataFrame."""
    rows = session.query(TickerSnapshot).filter(
        TickerSnapshot.squeeze_occurred.isnot(None)
    ).all()

    if not rows:
        raise ValueError("No labeled data found. Run build_training_data.py first.")

    data = []
    for r in rows:
        row = {col: getattr(r, col) for col in FEATURE_COLS}
        row["squeeze_occurred"] = int(r.squeeze_occurred)
        row["ticker"] = r.ticker
        row["macro_regime"] = r.macro_regime
        data.append(row)

    df = pd.DataFrame(data)
    console.print(f"[cyan]Loaded {len(df)} training rows[/cyan]")
    console.print(f"  Squeeze rate: {df['squeeze_occurred'].mean():.1%}")
    console.print(f"  Tickers: {df['ticker'].nunique()}")
    return df


def build_model(model_type: str):
    """Return sklearn-compatible model."""
    if model_type == "xgboost" and HAS_XGB:
        clf = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        )
    else:
        if model_type == "xgboost":
            console.print("[yellow]XGBoost not installed, falling back to logistic[/yellow]")
        clf = LogisticRegression(
            C=0.5,
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])
    return pipeline


def get_feature_importance(pipeline, feature_cols: list) -> dict:
    """Extract feature importances from trained model."""
    clf = pipeline.named_steps["clf"]
    importance = {}

    if hasattr(clf, "feature_importances_"):
        for col, imp in zip(feature_cols, clf.feature_importances_):
            importance[col] = round(float(imp), 4)
    elif hasattr(clf, "coef_"):
        for col, coef in zip(feature_cols, clf.coef_[0]):
            importance[col] = round(float(abs(coef)), 4)

    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


def train(model_type: str = "xgboost", save: bool = True):
    engine = init_db()
    session = get_session(engine)

    df = load_training_data(session)

    X = df[FEATURE_COLS]
    y = df["squeeze_occurred"]

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    console.print(f"\n[bold]Training {model_type} model...[/bold]")
    console.print(f"  Train: {len(X_train)} samples | Test: {len(X_test)} samples")

    model = build_model(model_type)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    # Pretty metrics table
    table = Table(title="Model Performance")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("ROC-AUC", f"{auc:.4f}")
    table.add_row("Precision", f"{precision:.4f}")
    table.add_row("Recall", f"{recall:.4f}")
    console.print(table)

    # Feature importance
    importance = get_feature_importance(model, FEATURE_COLS)
    console.print("\n[bold]Top features:[/bold]")
    for feat, imp in list(importance.items())[:8]:
        bar = "█" * int(imp * 40)
        console.print(f"  {feat:<30} {bar} {imp:.4f}")

    if save:
        os.makedirs("models", exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        console.print(f"\n[green]Model saved to {MODEL_PATH}[/green]")

        # Log to DB
        version_tag = f"v{datetime.now().strftime('%Y%m%d_%H%M')}"
        mv = ModelVersion(
            version_tag=version_tag,
            trained_at=datetime.utcnow(),
            training_samples=len(X_train),
            test_auc=auc,
            test_precision=precision,
            test_recall=recall,
            feature_importance_json=json.dumps(importance),
            notes=f"model_type={model_type}",
        )
        session.add(mv)
        session.commit()
        console.print(f"[green]Version logged: {version_tag}[/green]")

    return model, importance


def load_model():
    """Load saved model from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model at {MODEL_PATH}. Run train_model.py first."
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_proba(features: dict) -> float:
    """
    Score a single ticker snapshot.
    features: dict with keys matching FEATURE_COLS
    Returns: float probability 0.0 - 1.0
    """
    model = load_model()
    row = pd.DataFrame([{col: features.get(col) for col in FEATURE_COLS}])
    prob = model.predict_proba(row)[0][1]
    return round(float(prob), 4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["xgboost", "logistic"], default="xgboost")
    parser.add_argument("--eval", action="store_true", help="Don't save model")
    args = parser.parse_args()

    train(model_type=args.model, save=not args.eval)
