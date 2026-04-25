import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn
import pickle
import os

from data_generator import generate_claims_data
from feature_engineering import engineer_all_features, get_feature_columns


def train_isolation_forest(X_train: pd.DataFrame, contamination: float = 0.08):
    """
    Train Isolation Forest for unsupervised anomaly detection.
    Useful when labelled fraud data is scarce — common in real healthcare settings.
    """
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_samples="auto",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled)
    return model, scaler


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Train XGBoost classifier for supervised fraud detection.
    Uses scale_pos_weight to handle class imbalance (fraud is rare).
    """
    fraud_count  = y_train.sum()
    legit_count  = len(y_train) - fraud_count
    scale_weight = legit_count / fraud_count

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_weight,
        use_label_encoder=False,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train)],
        verbose=False
    )
    return model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series,
                   model_name: str, scaler=None) -> dict:
    """Evaluate model and return performance metrics."""
    if scaler is not None:
        X_eval = scaler.transform(X_test)
        raw_preds = model.predict(X_eval)
        preds = (raw_preds == -1).astype(int)
        proba = model.score_samples(X_eval)
        proba_norm = 1 - (proba - proba.min()) / (proba.max() - proba.min())
    else:
        preds      = model.predict(X_test)
        proba_norm = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model":      model_name,
        "accuracy":   round(accuracy_score(y_test, preds), 4),
        "precision":  round(precision_score(y_test, preds, zero_division=0), 4),
        "recall":     round(recall_score(y_test, preds, zero_division=0), 4),
        "f1_score":   round(f1_score(y_test, preds, zero_division=0), 4),
        "auc_roc":    round(roc_auc_score(y_test, proba_norm), 4),
    }

    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if k != "model":
            print(f"  {k:12}: {v}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, preds, target_names=["Legitimate", "Fraud"]))

    return metrics


def run_full_pipeline():
    """
    Full training pipeline:
    1. Generate synthetic claims data
    2. Engineer features
    3. Train both models
    4. Evaluate and log with MLflow
    5. Save models to disk
    """
    print("Step 1: Generating synthetic healthcare claims data...")
    df = generate_claims_data(n_records=10000, fraud_rate=0.08)

    print("Step 2: Engineering features...")
    df = engineer_all_features(df)

    feature_cols = get_feature_columns()
    available    = [c for c in feature_cols if c in df.columns]
    X = df[available].fillna(0)
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nDataset: {len(df)} claims | {y.sum()} fraud ({y.mean()*100:.1f}%)")
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    os.makedirs("models", exist_ok=True)
    all_metrics = []

    mlflow.set_experiment("healthcare-fraud-detection")

    print("\nStep 3: Training Isolation Forest...")
    with mlflow.start_run(run_name="isolation_forest"):
        iso_model, iso_scaler = train_isolation_forest(X_train, contamination=0.08)
        iso_metrics = evaluate_model(
            iso_model, X_test, y_test,
            "Isolation Forest", scaler=iso_scaler
        )
        for k, v in iso_metrics.items():
            if k != "model":
                mlflow.log_metric(k, v)
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("contamination", 0.08)
        all_metrics.append(iso_metrics)

    with open("models/isolation_forest.pkl", "wb") as f:
        pickle.dump({"model": iso_model, "scaler": iso_scaler, "features": available}, f)

    print("\nStep 4: Training XGBoost Classifier...")
    with mlflow.start_run(run_name="xgboost_classifier"):
        xgb_model   = train_xgboost(X_train, y_train)
        xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
        for k, v in xgb_metrics.items():
            if k != "model":
                mlflow.log_metric(k, v)
        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("max_depth", 6)
        mlflow.sklearn.log_model(xgb_model, "xgboost_fraud_model")
        all_metrics.append(xgb_metrics)

    with open("models/xgboost_fraud.pkl", "wb") as f:
        pickle.dump({"model": xgb_model, "features": available}, f)

    print("\nStep 5: Saving feature list...")
    with open("models/feature_columns.pkl", "wb") as f:
        pickle.dump(available, f)

    print("\nAll models saved to models/")
    print("\nModel Comparison:")
    metrics_df = pd.DataFrame(all_metrics).set_index("model")
    print(metrics_df.to_string())

    return xgb_model, iso_model, iso_scaler, available


if __name__ == "__main__":
    run_full_pipeline()
