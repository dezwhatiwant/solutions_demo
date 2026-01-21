def recommend_suppliers(df, top_n=3):
    df = df.copy()

    df["score"] = (
        df["on_time_rate"] * 0.35
        + (1 - df["avg_defect_rate"]) * 0.25
        + (1 - df["risk_score"]) * 0.20
        + (1 / df["avg_unit_price"]) * 0.20
    )

    return df.head(10)

def recommend_suppliers_for_part(part_description, orders_df, supplier_features, top_n=3):
    part_description = part_description.lower()

    # Find suppliers who have delivered this part before
    relevant_orders = orders_df[
        orders_df["part_description"].str.lower().str.contains(part_description, na=False)
    ]

    if relevant_orders.empty:
        print("\n❌ No matching parts found.\n")
        return None

    suppliers = relevant_orders["supplier_clean"].unique()
    df = supplier_features[supplier_features["supplier_clean"].isin(suppliers)].copy()

    # Reuse your scoring logic
    df["score"] = (
        df["on_time_rate"] * 0.35
        + (1 - df["avg_defect_rate"]) * 0.25
        + (1 - df["risk_score"]) * 0.20
        + (1 / df["avg_unit_price"]) * 0.20
    )

    return df.sort_values("score", ascending=False).head(top_n)
