from textblob import TextBlob
from rapidfuzz import process, fuzz

def analyze_notes(filepath, supplier_list):
    supplier_sentiment = {s: [] for s in supplier_list}

    with open(filepath, "r") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue

            sentiment = TextBlob(text).sentiment.polarity

            # Match note to closest supplier name
            match, score, _ = process.extractOne(
                text, supplier_list, scorer=fuzz.token_sort_ratio
            )

            if score > 70:
                supplier_sentiment[match].append(sentiment)

    return {
        s: (sum(vals) / len(vals)) if vals else 0.0
        for s, vals in supplier_sentiment.items()
    }
