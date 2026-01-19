from ingest import load_data
from normalize_suppliers import normalize
from features import engineer_features
from notes_nlp import analyze_notes
from risk_model import train_risk_model
from recommender import recommend_suppliers
from dashboard import supplier_dashboard

orders, quality, rfqs = load_data()

# Normalize supplier names
mapping = normalize(orders["supplier_name"].unique())
orders["supplier_clean"] = orders["supplier_name"].map(mapping)
rfqs["supplier_clean"] = rfqs["supplier_name"].map(mapping)

supplier_features = engineer_features(orders, quality, rfqs)

# NLP signal
notes_sentiment = analyze_notes(
    "data/raw/supplier_notes.txt",
    supplier_features["supplier_clean"].tolist()
)

supplier_features, model = train_risk_model(supplier_features)

recommendations = recommend_suppliers(supplier_features)
print(recommendations)

supplier_dashboard(supplier_features)
