import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


FRAUD_KEYWORDS = [
    "services rendered", "as documented", "per protocol",
    "billed accordingly", "medically necessary", "treatment administered",
    "comprehensive evaluation completed", "standard protocol",
    "as billed", "diagnosis confirmed"
]

LEGITIMATE_KEYWORDS = [
    "patient presents", "prescribed", "follow-up", "vitals normal",
    "blood work ordered", "medication adjusted", "physical therapy",
    "counseling provided", "post-operative", "wound healing",
    "shortness of breath", "chronic", "acute", "annual exam"
]


def clean_clinical_note(text: str) -> str:
    """Clean and normalize clinical note text."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def extract_nlp_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract NLP features from clinical notes.
    Mirrors real NLP pipelines used on EHR data at
    major health insurance companies.
    """
    df = df.copy()

    df["clean_note"] = df["clinical_note"].apply(clean_clinical_note)
    df["note_length"] = df["clean_note"].apply(lambda x: len(x.split()))
    df["note_char_count"] = df["clean_note"].apply(len)

    df["fraud_keyword_count"] = df["clean_note"].apply(
        lambda x: sum(1 for kw in FRAUD_KEYWORDS if kw in x)
    )
    df["legit_keyword_count"] = df["clean_note"].apply(
        lambda x: sum(1 for kw in LEGITIMATE_KEYWORDS if kw in x)
    )

    df["keyword_fraud_ratio"] = (
        df["fraud_keyword_count"] /
        (df["legit_keyword_count"] + df["fraud_keyword_count"] + 1)
    )

    df["has_fraud_keywords"]  = (df["fraud_keyword_count"] > 0).astype(int)
    df["has_legit_keywords"]  = (df["legit_keyword_count"] > 0).astype(int)
    df["is_vague_note"]       = (df["note_length"] < 6).astype(int)
    df["is_template_note"]    = (df["fraud_keyword_count"] >= 2).astype(int)

    df["unique_words"] = df["clean_note"].apply(
        lambda x: len(set(x.split()))
    )
    df["lexical_diversity"] = df["unique_words"] / (df["note_length"] + 1)

    return df


def build_tfidf_features(df: pd.DataFrame,
                          max_features: int = 50,
                          n_components: int = 10) -> tuple:
    """
    Build TF-IDF features from clinical notes and reduce
    dimensions using SVD (Latent Semantic Analysis).

    Returns:
        tfidf_df: DataFrame with LSA topic features
        vectorizer: fitted TF-IDF vectorizer
        svd: fitted SVD model
    """
    notes = df["clean_note"].fillna("")

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        stop_words="english"
    )
    tfidf_matrix = vectorizer.fit_transform(notes)

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    lsa_matrix = svd.fit_transform(tfidf_matrix)

    col_names = [f"lsa_topic_{i}" for i in range(n_components)]
    tfidf_df  = pd.DataFrame(lsa_matrix, columns=col_names, index=df.index)

    print(f"TF-IDF variance explained: {svd.explained_variance_ratio_.sum():.3f}")
    return tfidf_df, vectorizer, svd


def get_nlp_feature_columns() -> list:
    """Return NLP feature column names for model input."""
    return [
        "note_length", "note_char_count",
        "fraud_keyword_count", "legit_keyword_count",
        "keyword_fraud_ratio", "has_fraud_keywords",
        "has_legit_keywords", "is_vague_note",
        "is_template_note", "unique_words", "lexical_diversity"
    ]


def analyze_note(note: str) -> dict:
    """
    Analyze a single clinical note and return NLP signals.
    Used by the FastAPI prediction endpoint.
    """
    clean = clean_clinical_note(note)
    words = clean.split()

    fraud_hits = [kw for kw in FRAUD_KEYWORDS if kw in clean]
    legit_hits = [kw for kw in LEGITIMATE_KEYWORDS if kw in clean]

    fraud_kw_count = len(fraud_hits)
    legit_kw_count = len(legit_hits)
    total = fraud_kw_count + legit_kw_count + 1

    return {
        "note_length":         len(words),
        "note_char_count":     len(clean),
        "fraud_keyword_count": fraud_kw_count,
        "legit_keyword_count": legit_kw_count,
        "keyword_fraud_ratio": fraud_kw_count / total,
        "has_fraud_keywords":  int(fraud_kw_count > 0),
        "has_legit_keywords":  int(legit_kw_count > 0),
        "is_vague_note":       int(len(words) < 6),
        "is_template_note":    int(fraud_kw_count >= 2),
        "unique_words":        len(set(words)),
        "lexical_diversity":   len(set(words)) / (len(words) + 1),
        "fraud_keywords_found": fraud_hits,
        "legit_keywords_found": legit_hits,
        "risk_signal": "High" if fraud_kw_count >= 2 else
                       "Medium" if fraud_kw_count == 1 else "Low"
    }
