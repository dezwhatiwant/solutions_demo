from rapidfuzz import process, fuzz

def normalize(names, threshold=85):
    canonical = []
    mapping = {}

    for name in names:
        if not canonical:
            canonical.append(name)
            mapping[name] = name
        else:
            match, score, _ = process.extractOne(
                name, canonical, scorer=fuzz.token_sort_ratio
            )
            if score >= threshold:
                mapping[name] = match
            else:
                canonical.append(name)
                mapping[name] = name

    return mapping
