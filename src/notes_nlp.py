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
        text = f.read()

    # Split by big separator lines
    blocks = re.split(r"={5,}", text)

    records = []

    for block in blocks:
        lines = block.strip().split("\n")
        if not lines:
            continue

        # First non-empty line is the supplier header
        header = lines[0].strip()
        if " - " not in header:
            continue

        supplier_raw = header.split(" - ")[0]
        supplier_clean = clean_name(supplier_raw)

        # Everything after header is sentiment text
        body = " ".join(lines[1:])
        body = re.sub(r"\s+", " ", body)

        if body.strip():
            sentiment = TextBlob(body).sentiment.polarity
            records.append((supplier_clean, sentiment))

    df = pd.DataFrame(records, columns=["supplier_clean", "sentiment"])

    return df.groupby("supplier_clean")["sentiment"].mean().to_dict()
