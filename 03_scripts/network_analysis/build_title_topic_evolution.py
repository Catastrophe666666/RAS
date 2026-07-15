from __future__ import annotations

import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(r"C:\ras_text_analysis")
EXPANDED_CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "namecorrected_lcsh_outputs" / "ras_subject_expanded.csv"
OUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "namecorrected_lcsh_outputs"
SUFFIX = sys.argv[3] if len(sys.argv) > 3 else "_namecorrected"
STOPWORDS_PATH = Path(sys.argv[4]) if len(sys.argv) > 4 else ROOT / "hit_stopwords.txt"

TERM_TIME_CSV = OUT_DIR / f"title_term_time_subject_expanded{SUFFIX}.csv"
BIGRAM_TIME_CSV = OUT_DIR / f"title_bigram_time_subject_expanded{SUFFIX}.csv"
TFIDF_TIME_CSV = OUT_DIR / f"title_tfidf_time_subject_expanded{SUFFIX}.csv"
TERM_SUBJECT_TIME_CSV = OUT_DIR / f"title_term_subject_time_subject_expanded{SUFFIX}.csv"
ARTICLE_TERMS_CSV = OUT_DIR / f"article_title_terms_subject_expanded{SUFFIX}.csv"


BASE_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "for",
    "from", "had", "has", "have", "he", "her", "his", "in", "into", "is",
    "it", "its", "of", "on", "or", "our", "she", "that", "the", "their",
    "them", "these", "this", "those", "to", "was", "were", "which", "with",
    "without", "within", "about", "after", "before", "between", "during",
    "under", "over", "through", "upon", "some", "new", "old", "present",
    "part", "parts", "no", "notes", "note", "notice", "notices", "remarks",
    "remark", "observations", "observation", "account", "accounts", "report",
    "reports", "contribution", "contributions", "memorandum", "lecture",
    "paper", "papers", "journal", "article", "articles", "translation",
    "translated", "extract", "extracts", "appendix", "letter", "letters",
    "miscellaneous", "review", "reviews", "chapter", "chapters", "volume",
    "vol", "series", "royal", "asiatic", "society", "china", "chinese",
}


def load_stopwords(path: Path) -> set[str]:
    words = set(BASE_STOPWORDS)
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            token = line.strip().lower()
            if token:
                words.add(token)
    return words


def normalize_title(title: str) -> str:
    title = str(title or "").lower()
    title = title.replace("yang - tsze", "yangtze")
    title = title.replace("yang-tsze", "yangtze")
    title = title.replace("ta - ts'ing", "qing")
    title = title.replace("ta-ts'ing", "qing")
    title = title.replace("meh tsi", "mozi")
    return title


def tokenize(title: str, stopwords: set[str]) -> list[str]:
    title = normalize_title(title)
    tokens = re.findall(r"[a-z][a-z'-]{1,}", title)
    clean = []
    for token in tokens:
        token = token.strip("-'")
        if len(token) < 3:
            continue
        if token in stopwords:
            continue
        if token.endswith("'s"):
            token = token[:-2]
        if token and token not in stopwords:
            clean.append(token)
    return clean


def bigrams(tokens: list[str]) -> list[str]:
    return [f"{a} {b}" for a, b in zip(tokens, tokens[1:]) if a != b]


def load_articles() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(EXPANDED_CSV, encoding="utf-8")
    df["year_start"] = pd.to_numeric(df["year_start"], errors="coerce")
    df = df[
        (df["subject_l1"] != "Unknown")
        & (df["subject_l2"] != "Unknown")
        & df["subject_l1"].notna()
        & df["subject_l2"].notna()
    ].copy()
    df = df.drop_duplicates(subset=["master_id", "subject_l2"])
    articles = df[
        ["master_id", "time_window", "author_raw", "title", "year_start"]
    ].drop_duplicates("master_id")
    article_subjects = df[
        ["master_id", "time_window", "subject_l1", "subject_l2"]
    ].drop_duplicates()
    return articles, article_subjects


def aggregate_counts(rows: list[dict], group_keys: list[str], value_key: str) -> pd.DataFrame:
    counter: Counter[tuple] = Counter()
    article_sets: dict[tuple, set[str]] = defaultdict(set)
    for row in rows:
        key = tuple(row[k] for k in group_keys) + (row[value_key],)
        counter[key] += row.get("count", 1)
        article_sets[key].add(row["master_id"])
    out = []
    for key, count in counter.items():
        record = {k: key[i] for i, k in enumerate(group_keys)}
        record[value_key] = key[-1]
        record["term_count"] = count
        record["article_count"] = len(article_sets[key])
        out.append(record)
    return pd.DataFrame(out)


def add_window_shares(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    totals = df.groupby(group_cols)["term_count"].sum().rename("window_term_total")
    df = df.merge(totals, on=group_cols, how="left")
    df["share"] = df["term_count"] / df["window_term_total"]
    return df.drop(columns=["window_term_total"])


def compute_tfidf(term_time: pd.DataFrame, value_col: str = "term") -> pd.DataFrame:
    windows = sorted(term_time["time_window"].unique())
    n_windows = len(windows)
    doc_freq = term_time.groupby(value_col)["time_window"].nunique().to_dict()
    out = term_time.copy()
    out["idf"] = out[value_col].map(lambda term: math.log((1 + n_windows) / (1 + doc_freq.get(term, 0))) + 1)
    out["tfidf"] = out["term_count"] * out["idf"]
    return out.sort_values(["time_window", "tfidf"], ascending=[True, False])


def build_title_topics() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stopwords = load_stopwords(STOPWORDS_PATH)
    articles, article_subjects = load_articles()

    article_term_rows = []
    term_rows = []
    bigram_rows = []
    for row in articles.to_dict("records"):
        tokens = tokenize(row["title"], stopwords)
        token_counts = Counter(tokens)
        bigram_counts = Counter(bigrams(tokens))
        for term, count in token_counts.items():
            base = {
                "master_id": row["master_id"],
                "time_window": row["time_window"],
                "author_raw": row["author_raw"],
                "title": row["title"],
                "year_start": row["year_start"],
                "term": term,
                "count": count,
            }
            term_rows.append(base)
            article_term_rows.append(base)
        for phrase, count in bigram_counts.items():
            bigram_rows.append(
                {
                    "master_id": row["master_id"],
                    "time_window": row["time_window"],
                    "bigram": phrase,
                    "count": count,
                }
            )

    term_time = aggregate_counts(term_rows, ["time_window"], "term")
    term_time = add_window_shares(term_time, ["time_window"]).sort_values(["time_window", "term_count"], ascending=[True, False])
    bigram_time = aggregate_counts(bigram_rows, ["time_window"], "bigram")
    bigram_time = add_window_shares(bigram_time, ["time_window"]).sort_values(["time_window", "term_count"], ascending=[True, False])
    tfidf = compute_tfidf(term_time, "term")

    term_subject_rows = []
    terms_df = pd.DataFrame(article_term_rows)
    if not terms_df.empty:
        terms_subject = terms_df.merge(article_subjects, on=["master_id", "time_window"], how="inner")
        for row in terms_subject.to_dict("records"):
            term_subject_rows.append(row)
    term_subject_time = aggregate_counts(term_subject_rows, ["time_window", "subject_l1", "subject_l2"], "term")
    if not term_subject_time.empty:
        term_subject_time = add_window_shares(term_subject_time, ["time_window", "subject_l1", "subject_l2"])
        term_subject_time = compute_tfidf(term_subject_time, "term")

    article_terms = pd.DataFrame(article_term_rows)
    if not article_terms.empty:
        article_terms = article_terms.drop(columns=["count"]).drop_duplicates()

    term_time.to_csv(TERM_TIME_CSV, index=False, encoding="utf-8")
    bigram_time.to_csv(BIGRAM_TIME_CSV, index=False, encoding="utf-8")
    tfidf.to_csv(TFIDF_TIME_CSV, index=False, encoding="utf-8")
    term_subject_time.to_csv(TERM_SUBJECT_TIME_CSV, index=False, encoding="utf-8")
    article_terms.to_csv(ARTICLE_TERMS_CSV, index=False, encoding="utf-8")

    print(f"articles={len(articles)}")
    print(f"term_time={len(term_time)} bigram_time={len(bigram_time)} tfidf={len(tfidf)} term_subject_time={len(term_subject_time)}")
    print(f"wrote {TERM_TIME_CSV}")


if __name__ == "__main__":
    build_title_topics()
