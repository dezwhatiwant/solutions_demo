from sklearn.ensemble import RandomForestClassifier
import numpy as np

def train_risk_model(df):
    feature_cols = [
        "on_time_rate",
        "avg_days_late",
        "avg_defect_rate",
        "rework_rate",
        "avg_unit_price",
    ]

    X = df[feature_cols].fillna(0)

    # 🔹 Softer + more realistic risk definition
    y = (
        (df["on_time_rate"] < 0.92) |
        (df["avg_days_late"] > 2) |
        (df["avg_defect_rate"] > 0.015) |
        (df["rework_rate"] > 0.03)
    ).astype(int)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        random_state=42,
    )

    model.fit(X, y)

    # 🔹 Handle single-class case
    if len(model.classes_) == 1:
        df["risk_score"] = 0.0 if model.classes_[0] == 0 else 1.0
    else:
        df["risk_score"] = model.predict_proba(X)[:, 1]

    return df, model
