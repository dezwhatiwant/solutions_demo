import pandas as pd
from ingest import load_data
from normalize_suppliers import normalize
from features import engineer_features
from notes_nlp import analyze_notes
from risk_model import train_risk_model
from recommender import recommend_suppliers
from recommender import recommend_suppliers_for_part
from dashboard import supplier_dashboard
from cli import analyze_supplier
from risk_model import compute_absolute_risk


orders, quality, rfqs = load_data()

# Normalize supplier names from orders + RFQs only
all_names = pd.concat([
    orders["supplier_name"],
    rfqs["supplier_name"]
]).dropna().unique()

mapping = normalize(all_names)

orders["supplier_clean"] = orders["supplier_name"].map(mapping)
rfqs["supplier_clean"] = rfqs["supplier_name"].map(mapping)

supplier_features = engineer_features(orders, quality, rfqs)

# NLP signal
notes_sentiment = analyze_notes("data/raw/supplier_notes.txt")
supplier_features["notes_sentiment"] = (
    supplier_features["supplier_clean"]
    .map(notes_sentiment)
    .fillna(0)
)

supplier_features, model = train_risk_model(supplier_features)
supplier_features = compute_absolute_risk(supplier_features)

# Blend NLP sentiment into absolute risk score
supplier_features["risk_score"] = (
    supplier_features["absolute_risk"]
    - supplier_features["notes_sentiment"] * 0.30
).clip(0, 1)


print(supplier_features[["supplier_clean", "notes_sentiment", "risk_score"]])

recommendations = recommend_suppliers(supplier_features)
print("\nSupplier Breakdown:\n")
print(recommendations)

supplier_dashboard(supplier_features)

while True:
    print("\nOptions:")
    print("1 - Analyze supplier risk")
    print("2 - Recommend top suppliers for a part")
    print("3 - Quit")

    choice = input("Select option: ").strip()

    if choice == "1":
        name = input("Enter supplier name: ")
        analyze_supplier(name, supplier_features)

    elif choice == "2":
        part = input("Enter part description (e.g. 'Heat Exchanger'): ")
        top = recommend_suppliers_for_part(part, orders, supplier_features)

        if top is not None:
            print(f"\n🔎 Top Suppliers for '{part.title()}'")
            print("-" * 50)
            print(top[["supplier_clean", "score", "risk_score", "on_time_rate", "avg_unit_price"]])

    elif choice == "3":
        print("Exiting...")
        break

    else:
        print("Invalid option. Try again.")

