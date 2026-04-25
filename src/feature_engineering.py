import pandas as pd
import numpy as np
from datetime import datetime


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract 20+ features from raw claims data.
    These features mirror what real healthcare fraud detection
    systems use in production at major insurers.
    """
    df = df.copy()
    df = add_billing_features(df)
    df = add_provider_features(df)
    df = add_patient_features(df)
    df = add_temporal_features(df)
    df = add_ratio_features(df)
    return df


def add_billing_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Billing pattern features — fraudsters typically overbill
    or bill for services not rendered.
    """
    # How much more than approved was claimed
    df["claim_to_approved_ratio"] = (
        df["claim_amount"] / df["approved_amount"].replace(0, np.nan)
    ).fillna(1.0)

    # Flag extreme overbilling (more than 3x approved)
    df["is_extreme_overbill"] = (df["claim_to_approved_ratio"] > 3.0).astype(int)

    # Amount denied by insurer
    df["denied_amount"] = df["claim_amount"] - df["approved_amount"]
    df["denial_rate"]   = df["denied_amount"] / df["claim_amount"].replace(0, np.nan)
    df["denial_rate"]   = df["denial_rate"].fillna(0).clip(0, 1)

    # Flag unusually high single claim
    mean_amt = df["claim_amount"].mean()
    std_amt  = df["claim_amount"].std()
    df["claim_zscore"]      = (df["claim_amount"] - mean_amt) / std_amt
    df["is_high_value_claim"] = (df["claim_zscore"] > 2.5).astype(int)

    return df


def add_provider_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Provider behavior features — fraudulent providers often
    submit unusually high volumes or amounts.
    """
    provider_stats = df.groupby("provider_id").agg(
        provider_avg_claim    = ("claim_amount", "mean"),
        provider_total_claims = ("claim_id",     "count"),
        provider_total_billed = ("claim_amount", "sum"),
        provider_fraud_rate   = ("is_fraud",     "mean") if "is_fraud" in df.columns else ("claim_amount", "count")
    ).reset_index()

    df = df.merge(provider_stats, on="provider_id", how="left")

    # How much does this provider's avg compare to overall avg
    overall_avg = df["claim_amount"].mean()
    df["provider_vs_avg_ratio"] = df["provider_avg_claim"] / overall_avg

    # Flag high-volume providers (potential mills)
    df["is_high_volume_provider"] = (df["provider_total_claims"] > 100).astype(int)

    # Flag providers billing much more than average
    df["is_high_billing_provider"] = (df["provider_vs_avg_ratio"] > 2.0).astype(int)

    return df


def add_patient_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Patient behavior features — fraud rings often use
    the same patients repeatedly or target vulnerable groups.
    """
    patient_stats = df.groupby("patient_id").agg(
        patient_total_claims  = ("claim_id",     "count"),
        patient_total_billed  = ("claim_amount", "sum"),
        patient_avg_claim     = ("claim_amount", "mean"),
        patient_unique_providers = ("provider_id", "nunique")
    ).reset_index()

    df = df.merge(patient_stats, on="patient_id", how="left")

    # High claim frequency patient (potential identity fraud)
    df["is_high_freq_patient"] = (df["patient_total_claims"] > 10).astype(int)

    # Patient seeing unusually many providers (doctor shopping)
    df["is_doctor_shopping"] = (df["patient_unique_providers"] > 5).astype(int)

    # Elderly patients are more vulnerable to fraud
    df["is_elderly"] = (df["patient_age"] > 65).astype(int)

    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Time-based features — fraud often clusters at
    certain times (end of quarter, weekends, etc.)
    """
    df["claim_date"] = pd.to_datetime(df["claim_date"])
    df["claim_month"]     = df["claim_date"].dt.month
    df["claim_dayofweek"] = df["claim_date"].dt.dayofweek
    df["claim_quarter"]   = df["claim_date"].dt.quarter

    # Weekend claims are unusual for non-emergency providers
    df["is_weekend_claim"] = (df["claim_dayofweek"] >= 5).astype(int)

    # End of quarter — some fraud spikes at reporting periods
    df["is_quarter_end"] = (df["claim_date"].dt.month.isin([3, 6, 9, 12])).astype(int)

    return df


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ratio features that capture unusual combinations
    of procedures, diagnoses, and hospital stays.
    """
    # Procedures per diagnosis — high ratio can indicate upcoding
    df["proc_per_diagnosis"] = (
        df["num_procedures"] / df["num_diagnoses"].replace(0, 1)
    )

    # Prior claims frequency — high frequency suggests fraud ring
    df["prior_claims_rate"] = df["prior_claims_30d"] / 30.0

    # Hospital days vs procedures ratio
    df["hosp_days_per_proc"] = (
        df["days_in_hospital"] / df["num_procedures"].replace(0, 1)
    )

    # Combined risk score (simple rule-based)
    df["rule_based_risk"] = (
        df["is_extreme_overbill"] * 3 +
        df["is_high_volume_provider"] * 2 +
        df["is_high_freq_patient"] * 2 +
        df["is_doctor_shopping"] * 2 +
        df["is_weekend_claim"] * 1 +
        df["is_high_value_claim"] * 2
    )

    return df


def get_feature_columns() -> list:
    """Return the list of features used for model training."""
    return [
        "claim_amount", "approved_amount", "patient_age",
        "num_procedures", "num_diagnoses", "days_in_hospital",
        "prior_claims_30d", "provider_claim_count",
        "claim_to_approved_ratio", "is_extreme_overbill",
        "denied_amount", "denial_rate", "claim_zscore",
        "is_high_value_claim", "provider_avg_claim",
        "provider_total_claims", "provider_vs_avg_ratio",
        "is_high_volume_provider", "is_high_billing_provider",
        "patient_total_claims", "patient_avg_claim",
        "patient_unique_providers", "is_high_freq_patient",
        "is_doctor_shopping", "is_elderly",
        "claim_month", "claim_dayofweek", "claim_quarter",
        "is_weekend_claim", "is_quarter_end",
        "proc_per_diagnosis", "prior_claims_rate",
        "hosp_days_per_proc", "rule_based_risk"
    ]
