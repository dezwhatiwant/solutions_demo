from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

def train_risk_model(df):
    features = [
        "on_time_rate",
        "avg_days_late",
        "avg_defect_rate",
        "rework_rate",
        "avg_unit_price",
        "order_count"
    ]

    X = df[features].fillna(0)

    # Create synthetic labels: high risk if late + defects + rework are bad
    df["risk_label"] = (
        (df["on_time_rate"] < 0.85).astype(int) +
        (df["avg_days_late"] > 2).astype(int) +
        (df["avg_defect_rate"] > 0.04).astype(int) +
        (df["rework_rate"] > 0.45).astype(int)
    ) >= 2

    y = df["risk_label"].astype(int)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    probs = model.predict_proba(X)[:, 1]

    scaler = MinMaxScaler()
    df["risk_score"] = scaler.fit_transform(probs.reshape(-1, 1))

    return df, model
