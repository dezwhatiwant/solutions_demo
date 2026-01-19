from rapidfuzz import process, fuzz
import re

SUFFIXES = [
    r"\bINCORPORATED\b", r"\bINC\b", r"\bLLC\b", r"\bLTD\b",
    r"\bCO\b", r"\bCORP\b", r"\bCORPORATION\b", r"\bMFG\b",
    r"\bMANUFACTURING\b"
]

def _preclean(name: str) -> str:
    if not isinstance(name, str):
        return name

    name = name.upper()
    name = re.sub(r"[^\w\s]", "", name)

    for suf in SUFFIXES:
        name = re.sub(suf, "", name)

    name = re.sub(r"\s+", " ", name).strip()
    return name


def normalize(names, threshold=85):
    canonical = []
    mapping = {}

    for raw_name in names:
        cleaned = _preclean(raw_name)

        if not canonical:
            canonical.append(cleaned)
            mapping[raw_name] = cleaned
        else:
            match, score, _ = process.extractOne(
                cleaned, canonical, scorer=fuzz.token_sort_ratio
            )
            if score >= threshold:
                mapping[raw_name] = match
            else:
                canonical.append(cleaned)
                mapping[raw_name] = cleaned

    return mapping
