from textblob import TextBlob
import re

def analyze_notes(filepath):
    supplier_sentiment = {}

    with open(filepath, "r") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue

            sentiment = TextBlob(text).sentiment.polarity

            supplier = re.split(r"\s", text)[0]

            supplier_sentiment.setdefault(supplier, []).append(sentiment)

    return {
        s: sum(vals) / len(vals)
        for s, vals in supplier_sentiment.items()
    }
