import pandas as pd

def load_data():
    orders = pd.read_csv("data/raw/supplier_orders.csv")
    quality = pd.read_csv("data/raw/quality_inspections.csv")
    rfqs = pd.read_csv("data/raw/rfq_responses.csv")

return orders, quality, rfqs
