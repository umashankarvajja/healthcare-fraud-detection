import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


def generate_claims_data(n_records: int = 10000, fraud_rate: float = 0.08) -> pd.DataFrame:
    """
    Generate realistic synthetic healthcare insurance claims data.
    
    Features mirror real claims datasets used in healthcare fraud detection:
    - Patient demographics
    - Provider information
    - Diagnosis and procedure codes
    - Billing amounts
    - Clinical notes
    """
    np.random.seed(42)
    random.seed(42)

    n_fraud = int(n_records * fraud_rate)
    n_legit = n_records - n_fraud

    diagnosis_codes = [
        "E11.9",  # Type 2 diabetes
        "I10",    # Hypertension
        "J06.9",  # Upper respiratory infection
        "M54.5",  # Low back pain
        "F32.9",  # Major depressive disorder
        "Z00.00", # General health exam
        "K21.0",  # GERD
        "G43.909",# Migraine
        "J45.909",# Asthma
        "N39.0",  # Urinary tract infection
    ]

    procedure_codes = [
        "99213", "99214", "99215",  # Office visits
        "93000", "93010",           # EKG
        "71046",                     # Chest X-ray
        "80053",                     # Comprehensive metabolic panel
        "85025",                     # CBC
        "99283", "99284",           # ED visits
        "27447",                     # Knee replacement
    ]

    specialties = [
        "General Practice", "Cardiology", "Orthopedics",
        "Neurology", "Psychiatry", "Emergency Medicine",
        "Internal Medicine", "Oncology"
    ]

    legitimate_notes = [
        "Patient presents with chronic back pain, prescribed NSAIDs and physical therapy.",
        "Routine annual exam, all vitals normal, blood work ordered.",
        "Follow-up for hypertension management, BP 138/88, medication adjusted.",
        "Patient reports fatigue and shortness of breath, ECG ordered.",
        "Diabetes management visit, HbA1c 7.2%, diet counseling provided.",
        "Acute respiratory infection, prescribed antibiotics for 7 days.",
        "Post-operative follow-up, wound healing well, no complications.",
        "Mental health evaluation, patient reports improved mood on medication.",
    ]

    fraudulent_notes = [
        "Services rendered as documented.",
        "Patient seen and treated per standard protocol.",
        "Comprehensive evaluation completed.",
        "All services medically necessary and provided.",
        "Treatment administered as billed.",
        "Patient examined, diagnosis confirmed, billed accordingly.",
    ]

    def generate_legitimate_claim():
        diag = random.choice(diagnosis_codes)
        proc = random.choice(procedure_codes)
        specialty = random.choice(specialties)
        base_amount = np.random.normal(350, 150)
        return {
            "claim_id":            f"CLM{np.random.randint(100000, 999999)}",
            "patient_id":          f"PAT{np.random.randint(10000, 99999)}",
            "provider_id":         f"PRV{np.random.randint(1000, 5000)}",
            "provider_specialty":  specialty,
            "diagnosis_code":      diag,
            "procedure_code":      proc,
            "claim_amount":        round(max(50, base_amount), 2),
            "approved_amount":     round(max(40, base_amount * np.random.uniform(0.75, 0.95)), 2),
            "patient_age":         int(np.random.normal(52, 18)),
            "num_procedures":      int(np.random.poisson(1.5)),
            "num_diagnoses":       int(np.random.poisson(1.8)),
            "days_in_hospital":    int(np.random.poisson(0.8)),
            "prior_claims_30d":    int(np.random.poisson(1.2)),
            "provider_claim_count":int(np.random.normal(45, 20)),
            "claim_date":          datetime(2023, 1, 1) + timedelta(days=np.random.randint(0, 548)),
            "clinical_note":       random.choice(legitimate_notes),
            "is_fraud":            0
        }

    def generate_fraudulent_claim():
        diag = random.choice(diagnosis_codes)
        proc = random.choice(procedure_codes)
        specialty = random.choice(specialties)
        base_amount = np.random.normal(350, 150)
        fraud_multiplier = np.random.uniform(2.5, 6.0)
        return {
            "claim_id":            f"CLM{np.random.randint(100000, 999999)}",
            "patient_id":          f"PAT{np.random.randint(10000, 99999)}",
            "provider_id":         f"PRV{np.random.randint(1000, 1200)}",
            "provider_specialty":  specialty,
            "diagnosis_code":      diag,
            "procedure_code":      proc,
            "claim_amount":        round(base_amount * fraud_multiplier, 2),
            "approved_amount":     round(base_amount * np.random.uniform(0.75, 0.95), 2),
            "patient_age":         int(np.random.normal(52, 18)),
            "num_procedures":      int(np.random.poisson(6.0)),
            "num_diagnoses":       int(np.random.poisson(5.0)),
            "days_in_hospital":    int(np.random.poisson(0.2)),
            "prior_claims_30d":    int(np.random.poisson(8.0)),
            "provider_claim_count":int(np.random.normal(180, 40)),
            "claim_date":          datetime(2023, 1, 1) + timedelta(days=np.random.randint(0, 548)),
            "clinical_note":       random.choice(fraudulent_notes),
            "is_fraud":            1
        }

    legit_claims  = [generate_legitimate_claim() for _ in range(n_legit)]
    fraud_claims  = [generate_fraudulent_claim() for _ in range(n_fraud)]
    all_claims    = legit_claims + fraud_claims

    df = pd.DataFrame(all_claims).sample(frac=1, random_state=42).reset_index(drop=True)
    df["patient_age"] = df["patient_age"].clip(18, 95)
    df["num_procedures"] = df["num_procedures"].clip(1, 20)
    df["num_diagnoses"] = df["num_diagnoses"].clip(1, 15)
    df["days_in_hospital"] = df["days_in_hospital"].clip(0, 30)
    df["prior_claims_30d"] = df["prior_claims_30d"].clip(0, 25)

    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    """Print a summary of the generated dataset."""
    return {
        "total_claims":     len(df),
        "fraud_claims":     df["is_fraud"].sum(),
        "legit_claims":     (df["is_fraud"] == 0).sum(),
        "fraud_rate_pct":   round(df["is_fraud"].mean() * 100, 2),
        "avg_claim_amount": round(df["claim_amount"].mean(), 2),
        "avg_fraud_amount": round(df[df["is_fraud"] == 1]["claim_amount"].mean(), 2),
        "avg_legit_amount": round(df[df["is_fraud"] == 0]["claim_amount"].mean(), 2),
        "unique_providers": df["provider_id"].nunique(),
        "unique_patients":  df["patient_id"].nunique(),
        "date_range":       f"{df['claim_date'].min().date()} to {df['claim_date'].max().date()}"
    }


if __name__ == "__main__":
    print("Generating synthetic healthcare claims data...")
    df = generate_claims_data(n_records=10000, fraud_rate=0.08)
    summary = get_data_summary(df)
    print("\nDataset Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    df.to_csv("data/claims.csv", index=False)
    print("\nData saved to data/claims.csv")
