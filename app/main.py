from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
import pickle
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from nlp_processor import analyze_note
from feature_engineering import get_feature_columns

app = FastAPI(
    title="Healthcare Claims Fraud Detection API",
    description="Detects fraudulent healthcare insurance claims using XGBoost + NLP.",
    version="1.0.0"
)

# ── Load models on startup ────────────────────────────────────
xgb_data = None
iso_data  = None

def load_models():
    global xgb_data, iso_data
    try:
        with open("models/xgboost_fraud.pkl", "rb") as f:
            xgb_data = pickle.load(f)
        with open("models/isolation_forest.pkl", "rb") as f:
            iso_data = pickle.load(f)
        print("Models loaded successfully.")
    except FileNotFoundError:
        print("Models not found. Run src/fraud_detector.py first.")

load_models()

# ── Request/Response schemas ──────────────────────────────────
class ClaimRequest(BaseModel):
    claim_amount:          float
    approved_amount:       float
    patient_age:           int
    num_procedures:        int
    num_diagnoses:         int
    days_in_hospital:      int
    prior_claims_30d:      int
    provider_claim_count:  int
    clinical_note:         Optional[str] = ""

class FraudPrediction(BaseModel):
    claim_id:              Optional[str] = "N/A"
    fraud_prediction:      str
    fraud_probability:     float
    risk_level:            str
    xgboost_verdict:       str
    isolation_forest_flag: str
    nlp_risk_signal:       str
    top_fraud_driver:      str
    recommendation:        str

class BatchRequest(BaseModel):
    claims: List[ClaimRequest]

# ── Helper functions ──────────────────────────────────────────
def build_feature_row(claim: ClaimRequest) -> pd.DataFrame:
    """Convert a claim request into model-ready features."""
    row = {
        "claim_amount":         claim.claim_amount,
        "approved_amount":      claim.approved_amount,
        "patient_age":          claim.patient_age,
        "num_procedures":       claim.num_procedures,
        "num_diagnoses":        claim.num_diagnoses,
        "days_in_hospital":     claim.days_in_hospital,
        "prior_claims_30d":     claim.prior_claims_30d,
        "provider_claim_count": claim.provider_claim_count,
    }

    # Derived features
    ratio = claim.claim_amount / max(claim.approved_amount, 1)
    row["claim_to_approved_ratio"] = ratio
    row["is_extreme_overbill"]     = int(ratio > 3.0)
    row["denied_amount"]           = claim.claim_amount - claim.approved_amount
    row["denial_rate"]             = max(0, min(1, (claim.claim_amount - claim.approved_amount) / max(claim.claim_amount, 1)))
    row["claim_zscore"]            = 0.0
    row["is_high_value_claim"]     = int(claim.claim_amount > 1500)
    row["provider_avg_claim"]      = claim.claim_amount
    row["provider_total_claims"]   = claim.provider_claim_count
    row["provider_total_billed"]   = claim.claim_amount * claim.provider_claim_count
    row["provider_vs_avg_ratio"]   = claim.claim_amount / 350
    row["is_high_volume_provider"] = int(claim.provider_claim_count > 100)
    row["is_high_billing_provider"]= int(ratio > 2.0)
    row["patient_total_claims"]    = claim.prior_claims_30d
    row["patient_total_billed"]    = claim.claim_amount * claim.prior_claims_30d
    row["patient_avg_claim"]       = claim.claim_amount
    row["patient_unique_providers"]= 1
    row["is_high_freq_patient"]    = int(claim.prior_claims_30d > 10)
    row["is_doctor_shopping"]      = int(claim.prior_claims_30d > 8)
    row["is_elderly"]              = int(claim.patient_age > 65)
    row["claim_month"]             = 1
    row["claim_dayofweek"]         = 0
    row["claim_quarter"]           = 1
    row["is_weekend_claim"]        = 0
    row["is_quarter_end"]          = 0
    row["proc_per_diagnosis"]      = claim.num_procedures / max(claim.num_diagnoses, 1)
    row["prior_claims_rate"]       = claim.prior_claims_30d / 30.0
    row["hosp_days_per_proc"]      = claim.days_in_hospital / max(claim.num_procedures, 1)
    row["rule_based_risk"]         = (
        row["is_extreme_overbill"] * 3 +
        row["is_high_volume_provider"] * 2 +
        row["is_high_freq_patient"] * 2 +
        row["is_doctor_shopping"] * 2 +
        row["is_high_value_claim"] * 2
    )

    return pd.DataFrame([row])


def get_risk_level(probability: float) -> str:
    if probability >= 0.75:
        return "Critical Risk"
    elif probability >= 0.50:
        return "High Risk"
    elif probability >= 0.30:
        return "Medium Risk"
    else:
        return "Low Risk"


def get_recommendation(probability: float, risk: str) -> str:
    if probability >= 0.75:
        return "REJECT — Flag for immediate manual review and investigation."
    elif probability >= 0.50:
        return "HOLD — Requires secondary review before approval."
    elif probability >= 0.30:
        return "REVIEW — Approve with additional documentation request."
    else:
        return "APPROVE — Claim appears legitimate."


# ── Endpoints ─────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service":     "Healthcare Claims Fraud Detection API",
        "version":     "1.0.0",
        "status":      "running",
        "models_loaded": xgb_data is not None,
        "endpoints":   ["/predict", "/predict/batch", "/health", "/docs"]
    }


@app.get("/health")
def health():
    return {
        "status":          "healthy",
        "xgboost_loaded":  xgb_data is not None,
        "iso_forest_loaded": iso_data is not None
    }


@app.post("/predict", response_model=FraudPrediction)
def predict_fraud(claim: ClaimRequest):
    if xgb_data is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Run src/fraud_detector.py first."
        )

    features   = xgb_data["features"]
    xgb_model  = xgb_data["model"]
    iso_model  = iso_data["model"]
    iso_scaler = iso_data["scaler"]

    X = build_feature_row(claim)
    available = [c for c in features if c in X.columns]
    X_input   = X[available].fillna(0)

    xgb_prob  = float(xgb_model.predict_proba(X_input)[0][1])
    xgb_pred  = xgb_model.predict(X_input)[0]

    X_scaled  = iso_scaler.transform(X_input)
    iso_pred  = iso_model.predict(X_scaled)[0]

    nlp_result = analyze_note(claim.clinical_note or "")

    risk_level     = get_risk_level(xgb_prob)
    recommendation = get_recommendation(xgb_prob, risk_level)

    feature_importance = dict(zip(
        xgb_data["features"],
        xgb_model.feature_importances_
    ))
    top_driver = max(
        [f for f in available if f in feature_importance],
        key=lambda f: feature_importance.get(f, 0),
        default="N/A"
    )

    return FraudPrediction(
        fraud_prediction      = "FRAUD" if xgb_pred == 1 else "LEGITIMATE",
        fraud_probability     = round(xgb_prob, 4),
        risk_level            = risk_level,
        xgboost_verdict       = "Fraud" if xgb_pred == 1 else "Legitimate",
        isolation_forest_flag = "Anomaly" if iso_pred == -1 else "Normal",
        nlp_risk_signal       = nlp_result.get("risk_signal", "N/A"),
        top_fraud_driver      = top_driver.replace("_", " ").title(),
        recommendation        = recommendation
    )


@app.post("/predict/batch")
def predict_batch(batch: BatchRequest):
    if xgb_data is None:
        raise HTTPException(status_code=503, detail="Models not loaded.")

    results = []
    for i, claim in enumerate(batch.claims):
        try:
            result = predict_fraud(claim)
            result.claim_id = f"CLAIM_{i+1}"
            results.append(result)
        except Exception as e:
            results.append({"claim_id": f"CLAIM_{i+1}", "error": str(e)})

    fraud_count = sum(1 for r in results if hasattr(r, "fraud_prediction") and r.fraud_prediction == "FRAUD")
    return {
        "total_claims":    len(results),
        "fraud_detected":  fraud_count,
        "fraud_rate_pct":  round(fraud_count / len(results) * 100, 2),
        "predictions":     results
    }


@app.get("/example")
def example():
    return {
        "legitimate_claim": {
            "claim_amount": 320.00,
            "approved_amount": 290.00,
            "patient_age": 54,
            "num_procedures": 2,
            "num_diagnoses": 2,
            "days_in_hospital": 0,
            "prior_claims_30d": 1,
            "provider_claim_count": 40,
            "clinical_note": "Patient presents with chronic back pain, prescribed NSAIDs and physical therapy."
        },
        "suspicious_claim": {
            "claim_amount": 4200.00,
            "approved_amount": 290.00,
            "patient_age": 72,
            "num_procedures": 12,
            "num_diagnoses": 9,
            "days_in_hospital": 0,
            "prior_claims_30d": 18,
            "provider_claim_count": 210,
            "clinical_note": "Services rendered as documented. Treatment administered as billed."
        }
    }
