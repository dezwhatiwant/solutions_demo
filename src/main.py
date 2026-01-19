from ingest import load_data
from normalize_suppliers import normalize
from features import engineer_features
from notes_nlp import analyze_notes
from risk_model import train_risk_model
from recommender import recommend_suppliers
from dashboard import supplier_dashboard
import pandas as pd

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
notes_sentiment = analyze_notes(
    "data/raw/supplier_notes.txt",
    supplier_features["supplier_clean"].tolist()
)

supplier_features["notes_sentiment"] = (
    supplier_features["supplier_clean"]
    .map(notes_sentiment)
    .fillna(0)
    )

supplier_features, model = train_risk_model(supplier_features)

supplier_features["risk_score"] = (
    supplier_features["risk_score"]
    - supplier_features["notes_sentiment"] * 0.15
).clip(0, 1)

recommendations = recommend_suppliers(supplier_features)
print("\nTop Supplier Recommendations:\n")
print(recommendations)

supplier_dashboard(supplier_features)
