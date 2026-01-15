from sklearn.ensemble import RandomForestClassifier

def train_risk_model(df):
    feature_cols = [
        "on_time_rate",
        "avg_days_late",
        "avg_defect_rate",
        "rework_rate",
        "avg_unit_price",
    ]

    X = df[feature_cols].fillna(0)

    # Risk definition: suppliers with chronic lateness OR quality issues
    y = (
        (df["on_time_rate"] < 0.85)
        | (df["avg_defect_rate"] > 0.02)
        | (df["rework_rate"] > 0.05)
    ).astype(int)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=42,
    )
    model.fit(X, y)

    df["risk_score"] = model.predict_proba(X)[:, 1]

    return df, model
