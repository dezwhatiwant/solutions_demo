from textblob import TextBlob
import pandas as pd
import re

def clean_name(name):
    name = name.upper()
    name = re.sub(r"[^\w\s]", "", name)
    name = name.replace(" INC", "").replace(" LLC", "").replace(" MFG", "")
    name = re.sub(r"\s+", " ", name).strip()
    return name


def analyze_notes(notes_file):
    with open(notes_file, "r") as f:
        lines = f.readlines()

    records = []
    for line in lines:
        if ":" in line:
            supplier, note = line.split(":", 1)
            supplier_clean = clean_name(supplier)
            sentiment = TextBlob(note).sentiment.polarity
            records.append((supplier_clean, sentiment))

    df = pd.DataFrame(records, columns=["supplier_clean", "sentiment"])
    return df.groupby("supplier_clean")["sentiment"].mean().to_dict()
