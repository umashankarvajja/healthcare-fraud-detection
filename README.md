# Healthcare Claims Fraud Detection System

A production-grade ML system that detects fraudulent healthcare insurance claims
using anomaly detection, supervised learning, NLP on clinical notes, and SHAP
explainability — deployable on AWS SageMaker.

Directly inspired by real-world healthcare data science work in insurance and
clinical analytics environments.

## What This Project Does
- Generates realistic synthetic healthcare claims data (10,000+ records)
- Extracts 20+ engineered features from claims (billing patterns, provider behavior, diagnosis codes)
- Detects fraud using Isolation Forest (unsupervised) + XGBoost (supervised)
- Processes clinical notes using NLP (TF-IDF + keyword extraction)
- Explains model predictions using SHAP values
- Serves predictions via a FastAPI REST endpoint
- Fully deployable on AWS SageMaker

## Tech Stack
- Python, Pandas, NumPy
- Scikit-learn (Isolation Forest, preprocessing)
- XGBoost (supervised fraud classification)
- SHAP (model explainability)
- NLTK + Scikit-learn (NLP on clinical notes)
- FastAPI (REST API)
- MLflow (experiment tracking)
- Docker (containerization)
- AWS SageMaker (deployment)

## Project Structure
