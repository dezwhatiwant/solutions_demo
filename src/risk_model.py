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

    # 🔹 Peer-relative risk labeling
    on_time_thresh = df["on_time_rate"].quantile(0.25)
    defect_thresh = df["avg_defect_rate"].quantile(0.75)
    late_thresh = df["avg_days_late"].quantile(0.75)

    y = (
        (df["on_time_rate"] <= on_time_thresh) |
        (df["avg_defect_rate"] >= defect_thresh) |
        (df["avg_days_late"] >= late_thresh)
    ).astype(int)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        random_state=42,
    )
    model.fit(X, y)

    if len(model.classes_) == 1:
        df["risk_score"] = 0.0 if model.classes_[0] == 0 else 1.0
    else:
        df["risk_score"] = model.predict_proba(X)[:, 1]

    return df, model

def compute_absolute_risk(df):
    df = df.copy()

    # Convert metrics into 0–1 "badness" scores
    df["late_risk"] = (df["avg_days_late"] / 10).clip(0, 1)
    df["ontime_risk"] = (1 - df["on_time_rate"]).clip(0, 1)
    df["defect_risk"] = (df["avg_defect_rate"] / 0.05).clip(0, 1)
    df["rework_risk"] = (df["rework_rate"] / 0.5).clip(0, 1)

    # Weighted absolute risk
    df["absolute_risk"] = (
        df["ontime_risk"] * 0.35
        + df["late_risk"] * 0.25
        + df["defect_risk"] * 0.20
        + df["rework_risk"] * 0.20
    )

    return df

