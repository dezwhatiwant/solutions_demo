def analyze_supplier(supplier_name, df):
    supplier_name = supplier_name.upper().strip()
    row = df[df["supplier_clean"] == supplier_name]

    if row.empty:
        print(f"\n❌ Supplier '{supplier_name}' not found.\n")
        return

    r = row.iloc[0]

    print(f"\n📊 Risk Analysis for {supplier_name}")
    print("-" * 40)
    print(f"On-Time Delivery Rate: {r.on_time_rate:.2%}")
    print(f"Avg Days Late:        {r.avg_days_late:.2f}")
    print(f"Defect Rate:          {r.avg_defect_rate:.2%}")
    print(f"Rework Rate:          {r.rework_rate:.2%}")
    print(f"Avg Unit Price:      ${r.avg_unit_price:,.2f}")
    print(f"Orders Evaluated:     {int(r.order_count)}")
    print(f"Notes Sentiment:      {r.notes_sentiment:.3f}")
    print(f"\n⚠️  Risk Score:       {r.risk_score:.3f}")

    if r.risk_score > 0.75:
        print("🔴 HIGH RISK – Avoid for critical orders")
    elif r.risk_score > 0.34:
        print("🟠 MODERATE RISK – Use with caution")
    else:
        print("🟢 LOW RISK – Preferred supplier")
