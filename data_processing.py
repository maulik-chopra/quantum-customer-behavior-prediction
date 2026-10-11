"""Dataset loading and leakage-safe preparation for Telco churn data."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_CANDIDATES = ("churn", "exited", "target", "label")

def detect_target(frame):
    """Find a likely binary churn label; callers can choose another column."""
    lookup = {str(c).strip().lower(): c for c in frame.columns}
    return next((lookup[name] for name in TARGET_CANDIDATES if name in lookup), None)

def clean_target(series):
    """Map common yes/no churn labels into 1/0 and reject unknown labels."""
    text = series.astype("string").str.strip().str.lower()
    mapping = {"yes": 1, "true": 1, "1": 1, "churn": 1,
               "no": 0, "false": 0, "0": 0, "stay": 0, "retained": 0}
    values = text.map(mapping)
    if values.isna().any():
        raise ValueError("Target must contain recognizable yes/no or 1/0 labels.")
    return values.astype(int)

def make_demo_data(rows=850, seed=17):
    """Build clearly labeled synthetic Telco-like examples for demonstration."""
    import numpy as np
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 73, rows)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], rows, p=[.55,.25,.20])
    internet = rng.choice(["DSL", "Fiber optic", "No"], rows, p=[.35,.48,.17])
    monthly = np.clip(rng.normal(65, 23, rows), 18, 120).round(2)
    support = rng.choice(["Yes", "No"], rows)
    logit = 1.0 - tenure*.045 + (contract == "Month-to-month")*1.1 + (internet == "Fiber optic")*.35 + (support == "No")*.45 + (monthly > 85)*.35
    probability = 1 / (1 + np.exp(-logit))
    return pd.DataFrame({"customerID": [f"DEMO-{i:04d}" for i in range(rows)], "tenure": tenure,
       "Contract": contract, "InternetService": internet, "MonthlyCharges": monthly,
       "TechSupport": support, "PaymentMethod": rng.choice(["Electronic check", "Bank transfer", "Credit card"], rows),
       "Churn": np.where(rng.random(rows) < probability, "Yes", "No")})

def prepare_data(frame, target_column, test_size=.25, seed=17):
    """Split raw rows first, then fit imputers/encoders/scaler on training rows only."""
    if target_column not in frame or frame[target_column].nunique(dropna=True) != 2:
        raise ValueError("Choose a target column with exactly two classes.")
    labels = clean_target(frame[target_column])
    # Remove identifiers: they identify customers but do not describe behavior.
    features = frame.drop(columns=[target_column]).drop(columns=[c for c in frame if str(c).lower() in {"customerid", "id"}], errors="ignore")
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in features.columns if c not in numeric]
    transform = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ], remainder="drop")
    raw_train, raw_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=seed, stratify=labels)
    # Training-only fit prevents test-set statistics from leaking into the model.
    X_train = transform.fit_transform(raw_train)
    X_test = transform.transform(raw_test)
    names = transform.get_feature_names_out()
    return X_train, X_test, y_train, y_test, transform, names, raw_train, raw_test
