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
    current_supplier = None

    for line in lines:
        line = line.strip()

        # Detect supplier section headers
        header_match = re.match(r"^([A-Z\s/&]+)\s+-", line)
        if header_match:
            current_supplier = clean_name(header_match.group(1))
            continue

        # Skip metadata lines
        if not current_supplier or not line or line.startswith(("Email", "Meeting", "Graham", "Cody", "Sarah", "Mike")):
            continue

        # Sentiment on actual content lines
        blob = TextBlob(line)
        polarity = blob.sentiment.polarity

        records.append((current_supplier, polarity))

    df = pd.DataFrame(records, columns=["supplier_clean", "sentiment"])

    return df.groupby("supplier_clean")["sentiment"].mean().to_dict()
