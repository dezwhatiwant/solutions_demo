def recommend_suppliers(df, top_n=3):
    df = df.copy()

    df["score"] = (
        df["on_time_rate"] * 0.35
        + (1 - df["avg_defect_rate"]) * 0.25
        + (1 - df["risk_score"]) * 0.20
        + (1 / df["avg_unit_price"]) * 0.20
    )

    return df.sort_values("score", ascending=False).head(top_n)
