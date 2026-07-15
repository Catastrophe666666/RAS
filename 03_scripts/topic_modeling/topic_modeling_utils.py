"""Shared helpers for the RAS topic-modeling workflow.

The functions here intentionally read the existing conservative text pipeline
outputs instead of rebuilding cleaning or metadata matching.
"""

from __future__ import annotations

import csv
import importlib.util
import math
import re
from datetime import datetime
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_DIR / "outputs"
TOPIC_DIR = OUTPUTS_DIR / "topic_modeling"
DATA_DIR = TOPIC_DIR / "data"
MODELS_DIR = TOPIC_DIR / "models"
TABLES_DIR = TOPIC_DIR / "tables"
FIGURES_DIR = TOPIC_DIR / "figures"
REPORTS_DIR = TOPIC_DIR / "reports"

RANDOM_SEED = 42
MODEL_MIN_DF = 5
MODEL_MAX_DF = 0.60
MODEL_MAX_FEATURES = 12000
MODEL_NGRAM_RANGE = (1, 3)
MODEL_MIN_TOKENS = 120

MASTER_CANDIDATES = [
    OUTPUTS_DIR / "article_text_master_dataset_v2_header_removed.csv",
    OUTPUTS_DIR / "article_text_master_dataset.csv",
]

ANALYSIS_TEXT_DIR_CANDIDATES = [
    OUTPUTS_DIR / "analysis_texts_v2_header_removed",
    OUTPUTS_DIR / "cleaned_wordbag_text_header_removed",
    OUTPUTS_DIR / "analysis_texts",
    OUTPUTS_DIR / "cleaned_wordbag_text",
]

CLEANED_TEXT_DIR_CANDIDATES = [
    OUTPUTS_DIR / "cleaned_texts_v2_header_removed",
    OUTPUTS_DIR / "cleaned_readable_texts_header_removed",
    OUTPUTS_DIR / "cleaned_texts",
    OUTPUTS_DIR / "cleaned_readable_texts",
]

BASE_MODEL_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "also", "am", "an", "and", "any",
    "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few",
    "for", "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its",
    "itself", "just", "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of",
    "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own",
    "same", "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs",
    "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to",
    "too", "under", "until", "up", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "will", "with", "you", "your", "yours",
    "yourself", "yourselves",
    "article", "articles", "chapter", "figure", "figures", "journal", "page", "pages", "part",
    "plate", "plates", "proceedings", "review", "reviews", "section", "society", "vol", "volume",
}

RAS_DOMAIN_STOPWORDS = {
    "apr", "april", "aug", "august", "dec", "december", "feb", "february", "jan", "january",
    "july", "june", "mar", "march", "may", "nov", "november", "oct", "october", "sept",
    "september",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "called", "day", "days", "fig", "fol", "good", "great", "large", "man", "men", "name",
    "names", "note", "notes", "page", "paper", "said", "says", "seen", "small", "time",
    "times", "used", "well", "known", "work", "year", "years",
}

NON_ARTICLE_PATTERNS = [
    "misc", "proceedings", "review", "reviews", "recent books", "literary notes",
    "notes and queries", "queries", "appendix", "list",
]

METADATA_ALIASES = {
    "master_id": ["master_id", "matched_master_id"],
    "filename": ["text_file", "filename"],
    "title": ["title", "matched_title"],
    "author": ["author", "matched_author", "author_raw"],
    "year": ["year", "matched_year", "filename_year_hint"],
    "volume": ["volume"],
    "issue": ["issue", "number"],
    "decade": ["decade"],
    "L1": ["subject_l1_list", "subject_l1", "L1 subject", "L1"],
    "L2": ["subject_l2_list", "subject_l2", "L2 subject", "L2"],
    "language": ["detected_language", "language"],
}


def ensure_topic_dirs() -> None:
    for path in [DATA_DIR, MODELS_DIR, TABLES_DIR, FIGURES_DIR, REPORTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def first_existing(paths: Iterable[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("None of these paths exist: " + ", ".join(str(p) for p in paths))


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def has_package(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def dependency_report(required: list[str]) -> list[str]:
    return [name for name in required if not has_package(name)]


def normalize_stopword_entry(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def load_model_stopwords(extra_path: Path | None = None, include_general_english: bool = True) -> set[str]:
    stopwords = set(BASE_MODEL_STOPWORDS) | set(RAS_DOMAIN_STOPWORDS)
    path = extra_path or (PROJECT_DIR.parent / "hit_stopwords.txt")
    if path.exists():
        for line in read_text(path).splitlines():
            entry = normalize_stopword_entry(line)
            if entry:
                stopwords.add(entry)
                if " " in entry:
                    stopwords.add(entry.replace(" ", "_"))
        for token in re.findall(r"[A-Za-z][A-Za-z'-]{1,}", read_text(path).lower()):
            token = token.strip("-'")
            if 2 <= len(token) <= 40:
                stopwords.add(token)
    if include_general_english and has_package("sklearn"):
        try:
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

            stopwords.update(str(word).lower() for word in ENGLISH_STOP_WORDS)
        except Exception:
            pass
    return stopwords


def load_master_dataset() -> pd.DataFrame:
    path = first_existing(MASTER_CANDIDATES)
    return pd.read_csv(path)


def source_paths() -> dict[str, Path]:
    return {
        "master_dataset": first_existing(MASTER_CANDIDATES),
        "analysis_text_dir": first_existing(ANALYSIS_TEXT_DIR_CANDIDATES),
        "cleaned_text_dir": first_existing(CLEANED_TEXT_DIR_CANDIDATES),
    }


def first_value(row: pd.Series, aliases: list[str]) -> object:
    for col in aliases:
        if col in row.index:
            value = row.get(col)
            if pd.notna(value) and str(value).strip():
                return value
    return ""


def normalized_metadata_row(row: pd.Series) -> dict[str, object]:
    out = {name: first_value(row, aliases) for name, aliases in METADATA_ALIASES.items()}
    out["year"] = parse_year(out.get("year", ""))
    out["decade"] = parse_decade(out.get("decade", ""), out["year"])
    return out


def parse_year(value: object) -> int | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    match = re.search(r"(18|19)\d{2}", text)
    return int(match.group(0)) if match else None


def parse_decade(value: object, year: int | None) -> str:
    if pd.notna(value) and str(value).strip():
        text = str(value).strip()
        match = re.search(r"(18|19)\d0", text)
        if match:
            return match.group(0) + "s"
        if text.lower().endswith("s"):
            return text
    if year:
        return f"{year // 10 * 10}s"
    return ""


def split_subjects(value: object) -> list[str]:
    if pd.isna(value):
        return []
    parts = re.split(r"\s*\|\s*|\s*;\s*", str(value))
    return [p.strip() for p in parts if p.strip()]


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def simple_tokenize(text: str) -> list[str]:
    value = str(text)
    if re.fullmatch(r"[a-zA-Z_\s'-]*", value):
        return [token for token in value.split() if token]
    return tokenize_model_text(text)


def tokenize_model_text(text: str, stopwords: set[str] | None = None) -> list[str]:
    stopwords = stopwords or set()
    tokens = []
    for token in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", str(text).lower()):
        token = token.strip("-'")
        if len(token) < 3 or token in stopwords:
            continue
        if sum(ch.isalpha() for ch in token) < 3:
            continue
        tokens.append(token)
    return tokens


def add_ngrams(tokens: list[str], stopwords: set[str] | None = None, ngram_range: tuple[int, int] = MODEL_NGRAM_RANGE) -> list[str]:
    stopwords = stopwords or set()
    min_n, max_n = ngram_range
    out = list(tokens) if min_n <= 1 else []
    for n in range(max(2, min_n), max_n + 1):
        for idx in range(0, max(len(tokens) - n + 1, 0)):
            phrase_tokens = tokens[idx : idx + n]
            if any(token in stopwords for token in phrase_tokens):
                continue
            phrase = "_".join(phrase_tokens)
            if phrase not in stopwords:
                out.append(phrase)
    return out


def model_exclusion_reason(
    row: pd.Series,
    token_count: int,
    min_tokens: int = MODEL_MIN_TOKENS,
    exclude_non_articles: bool = True,
) -> str:
    reasons = []
    if token_count < min_tokens:
        reasons.append(f"short_after_stopwords_lt_{min_tokens}")
    if not exclude_non_articles:
        return ";".join(dict.fromkeys(reasons))
    haystack = " ".join(
        str(row.get(col, "")).lower()
        for col in ["filename", "title"]
        if pd.notna(row.get(col, ""))
    )
    for pattern in NON_ARTICLE_PATTERNS:
        if pattern in haystack:
            reasons.append(pattern.replace(" ", "_"))
    return ";".join(dict.fromkeys(reasons))


def document_frequency_vocabulary(
    tokenized_docs: list[list[str]],
    min_df: int = MODEL_MIN_DF,
    max_df: float = MODEL_MAX_DF,
    max_features: int | None = MODEL_MAX_FEATURES,
) -> tuple[set[str], pd.DataFrame]:
    doc_freq = Counter()
    raw_counts = Counter()
    for tokens in tokenized_docs:
        doc_freq.update(set(tokens))
        raw_counts.update(tokens)
    n_docs = len(tokenized_docs)
    max_doc_count = math.floor(max_df * n_docs) if isinstance(max_df, float) and max_df <= 1 else int(max_df)
    max_doc_count = max(max_doc_count, min_df)
    rows = []
    for term, df in doc_freq.items():
        rows.append(
            {
                "term": term,
                "raw_count": int(raw_counts[term]),
                "document_count": int(df),
                "document_frequency": float(df / n_docs) if n_docs else 0.0,
                "kept_by_df_filter": bool(df >= min_df and df <= max_doc_count),
            }
        )
    vocab_df = pd.DataFrame(rows)
    if vocab_df.empty:
        return set(), vocab_df
    kept = vocab_df[vocab_df["kept_by_df_filter"]].sort_values(
        ["raw_count", "document_count", "term"], ascending=[False, False, True]
    )
    if max_features:
        kept = kept.head(max_features)
        vocab_df["kept_by_df_filter"] = vocab_df["term"].isin(set(kept["term"]))
    return set(kept["term"]), vocab_df.sort_values(["kept_by_df_filter", "raw_count"], ascending=[False, False])


def build_model_documents(
    articles: pd.DataFrame,
    text_col: str = "analysis_text",
    min_df: int = MODEL_MIN_DF,
    max_df: float = MODEL_MAX_DF,
    max_features: int | None = MODEL_MAX_FEATURES,
    ngram_range: tuple[int, int] = MODEL_NGRAM_RANGE,
    min_tokens: int = MODEL_MIN_TOKENS,
    exclude_non_articles: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    stopwords = load_model_stopwords()
    source_texts = articles[text_col].fillna("").astype(str).tolist()
    base_tokenized = [tokenize_model_text(text, stopwords) for text in source_texts]
    tokenized = [add_ngrams(tokens, stopwords, ngram_range) for tokens in base_tokenized]
    vocab, vocab_df = document_frequency_vocabulary(tokenized, min_df=min_df, max_df=max_df, max_features=max_features)
    filtered = [[token for token in tokens if token in vocab] for tokens in tokenized]
    out = articles.copy()
    reasons = [
        model_exclusion_reason(row, len(tokens), min_tokens, exclude_non_articles)
        for (_, row), tokens in zip(articles.iterrows(), base_tokenized)
    ]
    out["model_exclusion_reason"] = reasons
    out["model_include"] = [not bool(reason) for reason in reasons]
    out["model_source_column"] = text_col
    out["model_text"] = [" ".join(tokens) for tokens in filtered]
    out["model_unigram_count_before_df"] = [len(tokens) for tokens in base_tokenized]
    out["model_token_count_before_df"] = [len(tokens) for tokens in tokenized]
    out["model_token_count_after_df"] = [len(tokens) for tokens in filtered]
    config = {
        "tokenizer": "tokenize_model_text_latin_alpha_len3_plus_ngrams",
        "stopword_count": len(stopwords),
        "min_df": min_df,
        "max_df": max_df,
        "max_features": max_features,
        "ngram_range": f"{ngram_range[0]}-{ngram_range[1]}",
        "min_tokens": min_tokens,
        "vocabulary_size": len(vocab),
        "source_column": text_col,
    }
    return out, vocab_df, config


def load_articles() -> pd.DataFrame:
    path = DATA_DIR / "topic_modeling_articles.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run build_topic_modeling_dataset.py first: {path}")
    return pd.read_csv(path)


def load_model_documents() -> pd.DataFrame:
    path = DATA_DIR / "topic_modeling_model_documents.csv"
    if not path.exists():
        articles = load_articles()
        docs, _, _ = build_model_documents(articles)
        return docs
    return pd.read_csv(path)


def load_chunks() -> pd.DataFrame:
    path = DATA_DIR / "topic_modeling_chunks.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run build_topic_modeling_dataset.py first: {path}")
    return pd.read_csv(path)


def top_terms_by_group(df: pd.DataFrame, group_col: str, text_col: str, top_n: int = 25) -> pd.DataFrame:
    rows = []
    for group_value, group in df.dropna(subset=[group_col]).groupby(group_col):
        terms = tfidf_term_table(group[text_col].fillna(""), top_n=top_n, include_counts=False)
        if terms.empty:
            continue
        for row in terms.itertuples(index=False):
            rows.append({group_col: group_value, "rank": row.rank, "term": row.term, "tfidf_score": row.tfidf_score})
    return pd.DataFrame(rows)


def tfidf_term_table(texts: Iterable[str], top_n: int = 100, include_counts: bool = True) -> pd.DataFrame:
    try:
        return sparse_tfidf_term_table(texts, top_n=top_n, include_counts=include_counts)
    except (ImportError, ModuleNotFoundError):
        return python_tfidf_term_table(texts, top_n=top_n, include_counts=include_counts)


def sparse_tfidf_term_table(texts: Iterable[str], top_n: int = 100, include_counts: bool = True) -> pd.DataFrame:
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer

    docs = [str(text) for text in texts]
    if not docs:
        return pd.DataFrame()

    vectorizer = CountVectorizer(tokenizer=simple_tokenize, lowercase=False, token_pattern=None)
    try:
        counts = vectorizer.fit_transform(docs)
    except ValueError:
        return pd.DataFrame()

    if counts.shape[1] == 0:
        return pd.DataFrame()

    doc_lengths = np.asarray(counts.sum(axis=1)).ravel()
    doc_lengths[doc_lengths == 0] = 1
    doc_freq = np.asarray((counts > 0).sum(axis=0)).ravel()
    idf = np.log((1 + counts.shape[0]) / (1 + doc_freq)) + 1
    totals = np.asarray(counts.multiply(1 / doc_lengths[:, None]).multiply(idf).sum(axis=0)).ravel()

    limit = min(top_n, totals.size)
    if limit == 0:
        return pd.DataFrame()
    top_idx = np.argpartition(-totals, limit - 1)[:limit]
    top_idx = top_idx[np.argsort(-totals[top_idx])]
    terms = vectorizer.get_feature_names_out()
    raw_counts = np.asarray(counts.sum(axis=0)).ravel() if include_counts else None

    rows = []
    for rank, idx in enumerate(top_idx, start=1):
        row = {"rank": rank, "term": terms[idx], "tfidf_score": float(totals[idx])}
        if include_counts:
            row["raw_count"] = int(raw_counts[idx])
            row["document_count"] = int(doc_freq[idx])
        rows.append(row)
    return pd.DataFrame(rows)


def python_tfidf_term_table(texts: Iterable[str], top_n: int = 100, include_counts: bool = True) -> pd.DataFrame:
    tokenized = [simple_tokenize(text) for text in texts]
    doc_freq = Counter()
    term_counts = []
    raw_counts = Counter()
    for tokens in tokenized:
        counts = Counter(tokens)
        term_counts.append(counts)
        raw_counts.update(counts)
        doc_freq.update(counts.keys())

    n_docs = max(len(tokenized), 1)
    totals = Counter()
    for counts in term_counts:
        total = sum(counts.values()) or 1
        for term, count in counts.items():
            tf = count / total
            idf = math.log((1 + n_docs) / (1 + doc_freq[term])) + 1
            totals[term] += tf * idf

    rows = []
    for rank, (term, score) in enumerate(totals.most_common(top_n), start=1):
        row = {"rank": rank, "term": term, "tfidf_score": score}
        if include_counts:
            row["raw_count"] = raw_counts[term]
            row["document_count"] = doc_freq[term]
        rows.append(row)
    return pd.DataFrame(rows)


def tfidf_scores(tokenized_docs: list[list[str]]) -> list[dict[str, float]]:
    doc_freq = Counter()
    term_counts = []
    for tokens in tokenized_docs:
        counts = Counter(tokens)
        term_counts.append(counts)
        doc_freq.update(counts.keys())
    n_docs = max(len(tokenized_docs), 1)
    out = []
    for counts in term_counts:
        total = sum(counts.values()) or 1
        doc_scores = {}
        for term, count in counts.items():
            tf = count / total
            idf = math.log((1 + n_docs) / (1 + doc_freq[term])) + 1
            doc_scores[term] = tf * idf
        out.append(doc_scores)
    return out


def overall_tfidf_terms(texts: Iterable[str], top_n: int = 100) -> pd.DataFrame:
    return tfidf_term_table(texts, top_n=top_n, include_counts=True)


def topic_diversity(topic_words: pd.DataFrame, word_col: str = "word") -> float:
    if topic_words.empty or word_col not in topic_words:
        return 0.0
    words = topic_words[word_col].dropna().astype(str).tolist()
    return len(set(words)) / max(len(words), 1)


def aggregate_topic_by_metadata(doc_topics: pd.DataFrame, metadata_col: str) -> pd.DataFrame:
    if doc_topics.empty or metadata_col not in doc_topics.columns:
        return pd.DataFrame()
    topic_cols = [c for c in doc_topics.columns if c.startswith("topic_")]
    if not topic_cols:
        return pd.DataFrame()
    grouped = doc_topics.groupby(metadata_col)[topic_cols].mean().reset_index()
    return grouped


def representative_documents(doc_topics: pd.DataFrame, topic_col: str, n: int = 5) -> str:
    if topic_col not in doc_topics:
        return ""
    label_cols = [c for c in ["title", "author", "year", "filename"] if c in doc_topics.columns]
    rows = doc_topics.sort_values(topic_col, ascending=False).head(n)
    reps = []
    for _, row in rows.iterrows():
        bits = [str(row[c]) for c in label_cols if pd.notna(row.get(c)) and str(row.get(c)).strip()]
        reps.append(" - ".join(bits))
    return " | ".join(reps)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback = path.with_name(f"{path.stem}_new_{timestamp}{path.suffix}")
        df.to_csv(fallback, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
        print(f"Warning: {path} is locked; wrote {fallback} instead.")
