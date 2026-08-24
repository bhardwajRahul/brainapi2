import math


def okapi_idf(n_docs: float, df: float) -> float:
    n_docs = max(float(n_docs), 0.0)
    df = max(float(df), 0.0)
    return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))


def okapi_bm25_term(
    tf: float,
    dl: float,
    avgdl: float,
    idf: float,
    *,
    k1: float = 1.2,
    b: float = 0.75,
) -> float:
    if tf <= 0 or idf == 0:
        return 0.0
    avgdl = avgdl if avgdl > 0 else 1.0
    dl = dl if dl > 0 else 1.0
    denom = tf + k1 * (1.0 - b + b * dl / avgdl)
    if denom == 0:
        return 0.0
    return idf * (tf * (k1 + 1.0)) / denom


def okapi_bm25_document(
    term_tfs: dict[str, float],
    *,
    dl: float,
    avgdl: float,
    n_docs: float,
    dfs: dict[str, float],
    k1: float = 1.2,
    b: float = 0.75,
) -> float:
    score = 0.0
    for term, tf in term_tfs.items():
        idf = okapi_idf(n_docs, dfs.get(term, 0.0))
        score += okapi_bm25_term(tf, dl, avgdl, idf, k1=k1, b=b)
    return score
