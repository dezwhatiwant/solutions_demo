import pandas as pd

def engineer_features(orders, quality, rfqs):
    # --- Date handling ---
    orders["promised_date"] = pd.to_datetime(orders["promised_date"], errors="coerce")
    orders["actual_delivery_date"] = pd.to_datetime(
        orders["actual_delivery_date"], errors="coerce"
    )

    orders["on_time"] = (
        orders["actual_delivery_date"] <= orders["promised_date"]
    ).astype(int)

    orders["days_late"] = (
        orders["actual_delivery_date"] - orders["promised_date"]
    ).dt.days.clip(lower=0)

    # --- Quality metrics ---
    quality["inspection_date"] = pd.to_datetime(
        quality["inspection_date"], errors="coerce"
    )

    quality["defect_rate"] = (
        quality["parts_rejected"] / quality["parts_inspected"]
    ).fillna(0)

    quality["quality_failure"] = (quality["parts_rejected"] > 0).astype(int)
    quality["rework_required"] = quality["rework_required"].map({"Yes": 1, "No": 0})

    quality_summary = (
        quality.groupby("order_id")
        .agg(
            avg_defect_rate=("defect_rate", "mean"),
            quality_failure_rate=("quality_failure", "mean"),
            rework_rate=("rework_required", "mean"),
        )
        .reset_index()
    )

    # --- Merge operational + quality ---
    df = orders.merge(quality_summary, on="order_id", how="left")

    # --- RFQ price intelligence ---
    rfq_summary = (
        rfqs.groupby(["supplier_clean", "part_description"])
        .agg(
            avg_quoted_price=("quoted_price", "mean"),
            avg_lead_time=("lead_time_weeks", "mean"),
        )
        .reset_index()
    )

    df = df.merge(
        rfq_summary,
        on=["supplier_clean", "part_description"],
        how="left",
    )

    # --- Supplier-level aggregation ---
    supplier_features = (
        df.groupby("supplier_clean")
        .agg(
            on_time_rate=("on_time", "mean"),
            avg_days_late=("days_late", "mean"),
            avg_defect_rate=("avg_defect_rate", "mean"),
            rework_rate=("rework_rate", "mean"),
            avg_unit_price=("unit_price", "mean"),
            order_count=("order_id", "count"),
        )
        .reset_index()
    )

    return supplier_features
