import pandas as pd
import numpy as np
import pickle
import shap


def load_xgboost_model(model_path: str = "models/xgboost_fraud.pkl") -> tuple:
    """Load the trained XGBoost model and feature list."""
    with open(model_path, "rb") as f:
        saved = pickle.load(f)
    return saved["model"], saved["features"]


def get_shap_explainer(model, X_train: pd.DataFrame):
    """
    Build a SHAP TreeExplainer for the XGBoost model.
    SHAP (SHapley Additive exPlanations) explains WHY
    the model made a specific prediction — critical for
    healthcare fraud where decisions must be auditable.
    """
    explainer = shap.TreeExplainer(model)
    return explainer


def explain_single_claim(explainer, X_single: pd.DataFrame,
                          feature_names: list) -> dict:
    """
    Explain a single claim prediction using SHAP values.

    Returns the top features driving the fraud prediction
    and their contribution direction (increases/decreases risk).
    """
    shap_values = explainer.shap_values(X_single)

    if isinstance(shap_values, list):
        shap_vals = shap_values[1][0]
    else:
        shap_vals = shap_values[0]

    feature_impacts = pd.DataFrame({
        "feature":    feature_names,
        "shap_value": shap_vals,
        "abs_impact": np.abs(shap_vals)
    }).sort_values("abs_impact", ascending=False)

    top_features = feature_impacts.head(10)

    explanation = []
    for _, row in top_features.iterrows():
        direction = "increases" if row["shap_value"] > 0 else "decreases"
        explanation.append({
            "feature":    row["feature"],
            "impact":     round(float(row["shap_value"]), 4),
            "direction":  direction,
            "abs_impact": round(float(row["abs_impact"]), 4)
        })

    return {
        "top_10_features":   explanation,
        "top_fraud_driver":  explanation[0]["feature"] if explanation else "N/A",
        "explanation_summary": generate_explanation_text(explanation[:3])
    }


def generate_explanation_text(top_features: list) -> str:
    """
    Generate a human-readable explanation of the fraud prediction.
    This mirrors what a real fraud analyst would see in a dashboard.
    """
    if not top_features:
        return "Insufficient data to explain prediction."

    lines = ["This claim was flagged as potentially fraudulent because:"]
    for i, feat in enumerate(top_features, 1):
        feat_name = feat["feature"].replace("_", " ").title()
        direction = feat["direction"]
        impact    = abs(feat["impact"])
        lines.append(
            f"  {i}. {feat_name} {direction} fraud risk "
            f"(impact score: {impact:.3f})"
        )
    return "\n".join(lines)


def get_global_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """
    Get global feature importance from the XGBoost model.
    Shows which features matter most across ALL predictions.
    """
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        "feature":    feature_names,
        "importance": importance
    }).sort_values("importance", ascending=False)

    importance_df["importance_pct"] = (
        importance_df["importance"] /
        importance_df["importance"].sum() * 100
    ).round(2)

    return importance_df


def explain_batch(explainer, X_batch: pd.DataFrame,
                  feature_names: list, top_n: int = 5) -> pd.DataFrame:
    """
    Explain predictions for a batch of claims.
    Returns a summary DataFrame with top fraud driver per claim.
    """
    shap_values = explainer.shap_values(X_batch)

    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    results = []
    for i in range(len(X_batch)):
        row_shap  = shap_vals[i]
        top_idx   = np.argsort(np.abs(row_shap))[::-1][:top_n]
        top_feats = [feature_names[j] for j in top_idx]
        top_vals  = [round(float(row_shap[j]), 4) for j in top_idx]

        results.append({
            "claim_index":       i,
            "top_fraud_driver":  top_feats[0] if top_feats else "N/A",
            "top_driver_impact": top_vals[0] if top_vals else 0,
            "top_5_features":    ", ".join(top_feats)
        })

    return pd.DataFrame(results)
