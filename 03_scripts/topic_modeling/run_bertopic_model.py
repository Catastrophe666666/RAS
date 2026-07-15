"""Run BERTopic on article, paragraph, or chunk-level RAS text.

This version uses a dual-text workflow:
- embeddings are computed from cleaned readable text that preserves context;
- topic words/c-TF-IDF are computed from a filtered representation text.
"""

from __future__ import annotations

import re
from html import escape
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from topic_modeling_utils import (
    DATA_DIR,
    FIGURES_DIR,
    OUTPUTS_DIR,
    REPORTS_DIR,
    TABLES_DIR,
    dependency_report,
    ensure_topic_dirs,
    has_package,
    load_articles,
    load_chunks,
    read_text,
    save_csv,
    simple_tokenize,
    write_markdown,
)


BERTOPIC_LEVEL = "chunk"  # options: "article", "paragraph", "chunk"
BERTOPIC_VARIANT = "chunk_dual_stopworded"
RUN_VARIANT_COMPARISON = False

MIN_WORDS = 60
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# BERTopic tuning for long OCR corpora. Increase min_cluster_size if topics are
# too fine-grained; lower it if too many chunks become outliers.
MIN_CLUSTER_SIZE = 60
MIN_SAMPLES = 10
UMAP_N_NEIGHBORS = 20
UMAP_N_COMPONENTS = 5
UMAP_MIN_DIST = 0.0

# c-TF-IDF / vectorizer controls.
NGRAM_RANGE = (1, 2)
MIN_DF = 5
MAX_DF = 0.60
TOKEN_PATTERN = r"(?u)\b[a-zA-Z][a-zA-Z'-]{2,}\b"
USE_BM25_WEIGHTING = True
REDUCE_FREQUENT_WORDS = True
RUN_OUTLIER_REDUCTION = True
BARCHART_TOP_N_TOPICS = 30
BARCHART_N_WORDS = 10
BARCHART_INCLUDE_ALL_TOPICS = False

VARIANTS = [
    {
        "name": "chunk_dual_stopworded",
        "level": "chunk",
        "english_only": False,
        "dual_text": True,
        "min_cluster_size": MIN_CLUSTER_SIZE,
        "min_samples": MIN_SAMPLES,
    },
    {
        "name": "paragraph_dual_stopworded",
        "level": "paragraph",
        "english_only": False,
        "dual_text": True,
        "min_cluster_size": 45,
        "min_samples": 8,
    },
    {
        "name": "chunk_english_only_stopworded",
        "level": "chunk",
        "english_only": True,
        "dual_text": True,
        "min_cluster_size": MIN_CLUSTER_SIZE,
        "min_samples": MIN_SAMPLES,
    },
    {
        "name": "chunk_default_like",
        "level": "chunk",
        "english_only": False,
        "dual_text": False,
        "min_cluster_size": MIN_CLUSTER_SIZE,
        "min_samples": MIN_SAMPLES,
    },
]


BASE_STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}

RAS_OCR_STOPWORDS = {
    "article",
    "articles",
    "chapter",
    "delivered",
    "fig",
    "figure",
    "fol",
    "ibid",
    "journal",
    "ltd",
    "mr",
    "mrs",
    "ms",
    "note",
    "notes",
    "page",
    "pages",
    "part",
    "plate",
    "president",
    "printed",
    "proceedings",
    "read",
    "rev",
    "royal",
    "said",
    "section",
    "series",
    "sir",
    "society",
    "vol",
    "volume",
    # Frequent OCR/binomial abbreviation noise seen in current BERTopic output.
    "cult",
    "fl",
    "hgk",
    "linn",
    "lour",
    "osb",
}


@dataclass
class BertopicRunResult:
    name: str
    level: str
    documents: int
    topic_count: int
    outlier_count: int
    outlier_share: float
    output_prefix: str


def load_custom_stopwords() -> set[str]:
    stopwords = set(BASE_STOPWORDS) | set(RAS_OCR_STOPWORDS)
    if has_package("sklearn"):
        try:
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

            stopwords.update(ENGLISH_STOP_WORDS)
        except Exception:
            pass
    source = OUTPUTS_DIR.parent / "hit_stopwords.txt"
    for token in simple_tokenize(read_text(source)):
        if token.isascii() and 2 <= len(token) <= 30:
            stopwords.add(token.lower())
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "bertopic_custom_stopwords.txt").write_text("\n".join(sorted(stopwords)), encoding="utf-8")
    return stopwords


def filtered_representation_text(text: str, stopwords: set[str]) -> str:
    tokens = []
    for token in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", str(text).lower()):
        token = token.strip("-'")
        if len(token) < 3 or token in stopwords:
            continue
        if token.isupper():
            token = token.lower()
        if sum(ch.isalpha() for ch in token) < 3:
            continue
        tokens.append(token)
    return " ".join(tokens)


def paragraph_records() -> pd.DataFrame:
    articles = load_articles()
    rows = []
    for _, row in articles.iterrows():
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", str(row.get("cleaned_full_text", ""))) if p.strip()]
        for idx, para in enumerate(paragraphs):
            if len(para.split()) < MIN_WORDS:
                continue
            rows.append(
                {
                    "chunk_id": f"{row.get('master_id')}_p{idx:04d}",
                    "master_id": row.get("master_id"),
                    "filename": row.get("filename"),
                    "chunk_index": idx,
                    "embedding_text": para,
                    "chunk_word_count": len(para.split()),
                    "title": row.get("title"),
                    "author": row.get("author"),
                    "year": row.get("year"),
                    "decade": row.get("decade"),
                    "L1": row.get("L1"),
                    "L2": row.get("L2"),
                    "language": row.get("language"),
                }
            )
    return pd.DataFrame(rows)


def load_bertopic_input(level: str, stopwords: set[str], dual_text: bool, english_only: bool) -> pd.DataFrame:
    if level == "chunk":
        df = load_chunks().rename(columns={"chunk_text": "embedding_text"})
    elif level == "paragraph":
        df = paragraph_records()
    elif level == "article":
        articles = load_articles()
        df = articles.rename(columns={"cleaned_full_text": "embedding_text"}).copy()
        df["chunk_id"] = df["master_id"].astype(str) + "_article"
        df["chunk_index"] = 0
        df["chunk_word_count"] = df["word_count"]
    else:
        raise ValueError("level must be one of: article, paragraph, chunk")

    if english_only and "language" in df.columns:
        df = df[df["language"].fillna("").astype(str).str.lower().isin(["en", "english"])].copy()

    df = df[df["embedding_text"].fillna("").astype(str).str.split().str.len() >= MIN_WORDS].reset_index(drop=True)
    if dual_text:
        df["representation_text"] = df["embedding_text"].map(lambda text: filtered_representation_text(text, stopwords))
    else:
        df["representation_text"] = df["embedding_text"].astype(str)
    df = df[df["representation_text"].fillna("").astype(str).str.split().str.len() >= 8].reset_index(drop=True)
    return df


def make_models(stopwords: set[str], min_cluster_size: int, min_samples: int):
    from bertopic.vectorizers import ClassTfidfTransformer
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer_model = CountVectorizer(
        stop_words=sorted(stopwords),
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        max_df=MAX_DF,
        token_pattern=TOKEN_PATTERN,
    )
    ctfidf_model = ClassTfidfTransformer(
        bm25_weighting=USE_BM25_WEIGHTING,
        reduce_frequent_words=REDUCE_FREQUENT_WORDS,
    )

    umap_model = None
    hdbscan_model = None
    if has_package("umap"):
        from umap import UMAP

        umap_model = UMAP(
            n_neighbors=UMAP_N_NEIGHBORS,
            n_components=UMAP_N_COMPONENTS,
            min_dist=UMAP_MIN_DIST,
            metric="cosine",
            random_state=42,
        )
    if has_package("hdbscan"):
        from hdbscan import HDBSCAN

        hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )
    return vectorizer_model, ctfidf_model, umap_model, hdbscan_model


def write_placeholder_report(missing: list[str]) -> None:
    write_markdown(
        REPORTS_DIR / "bertopic_report.md",
        [
            "# BERTopic Report",
            "",
            "BERTopic was not run because required packages are unavailable:",
            ", ".join(missing),
            "",
            "Install `bertopic`, `sentence-transformers`, `scikit-learn`, `umap-learn`, and `hdbscan` to run this optional step.",
        ],
    )


def normalized_score(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        return pd.Series([1.0 if hi > 0 else 0.0] * len(values), index=values.index)
    return (values - lo) / (hi - lo)


def topic_label(words_df: pd.DataFrame, topic_id: int, n: int = 8) -> str:
    if words_df.empty:
        return ""
    words = (
        words_df[(words_df["topic_id"] == topic_id) & (words_df["rank"] <= n)]
        .sort_values("rank")["word"]
        .astype(str)
        .tolist()
    )
    return ", ".join(words)


def interpretability_from_words(words_df: pd.DataFrame, topic_id: int) -> float:
    weights = pd.to_numeric(
        words_df[words_df["topic_id"] == topic_id].sort_values("rank")["weight"],
        errors="coerce",
    ).fillna(0.0).head(15)
    total = float(weights.sum())
    if total <= 0:
        return 0.0
    return float(weights.head(5).sum() / total)


def build_bertopic_interpretation_tables(df: pd.DataFrame, words_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = df[df["topic_id"] != -1].copy()
    total_chunks = max(len(valid), 1)
    total_articles = max(valid["master_id"].nunique(), 1) if "master_id" in valid.columns else 1
    size = valid.groupby("topic_id").size().rename("topic_size")
    article_coverage = valid.groupby("topic_id")["master_id"].nunique().rename("article_count") if "master_id" in valid.columns else size.rename("article_count")
    summary = pd.concat([size, article_coverage], axis=1).reset_index()
    summary["topic_size_share"] = summary["topic_size"] / total_chunks
    summary["article_coverage"] = summary["article_count"] / total_articles
    summary["interpretability"] = summary["topic_id"].map(lambda topic_id: interpretability_from_words(words_df, int(topic_id)))
    summary["topic_words"] = summary["topic_id"].map(lambda topic_id: topic_label(words_df, int(topic_id), 12))
    summary["coverage_score_norm"] = normalized_score(summary["article_coverage"])
    summary["size_score_norm"] = normalized_score(summary["topic_size"])
    summary["interpretability_score_norm"] = normalized_score(summary["interpretability"])
    summary["importance_score"] = (
        summary["coverage_score_norm"] + summary["size_score_norm"] + summary["interpretability_score_norm"]
    ) / 3
    summary = summary.sort_values(["importance_score", "article_coverage", "topic_size"], ascending=False).reset_index(drop=True)
    summary["importance_rank"] = range(1, len(summary) + 1)

    by_decade = (
        valid.groupby(["decade", "topic_id"])
        .agg(count=("topic_id", "size"), article_count=("master_id", "nunique"))
        .reset_index()
    )
    decade_totals = valid.groupby("decade").size().rename("decade_total").reset_index()
    topic_totals = valid.groupby("topic_id").size().rename("topic_total").reset_index()
    by_decade = by_decade.merge(decade_totals, on="decade", how="left").merge(topic_totals, on="topic_id", how="left")
    by_decade["share_of_decade"] = by_decade["count"] / by_decade["decade_total"].clip(lower=1)
    by_decade["share_of_topic"] = by_decade["count"] / by_decade["topic_total"].clip(lower=1)
    by_decade["topic_words"] = by_decade["topic_id"].map(lambda topic_id: topic_label(words_df, int(topic_id), 6))
    return summary, by_decade


def write_topic_heatmap_html(table: pd.DataFrame, summary: pd.DataFrame, path: Path, title: str, value_col: str = "share_of_decade") -> None:
    if table.empty:
        path.write_text(f"<html><body><h1>{escape(title)}</h1><p>No topic-time data.</p></body></html>", encoding="utf-8")
        return
    top_topics = summary.sort_values("importance_rank").head(30)["topic_id"].tolist()
    data = table[table["topic_id"].isin(top_topics)].copy()
    pivot = data.pivot_table(index="topic_id", columns="decade", values=value_col, aggfunc="sum", fill_value=0)
    max_value = float(pivot.to_numpy().max()) if pivot.size else 0.0
    labels = summary.set_index("topic_id")["topic_words"].to_dict()
    rows = []
    for topic_id, values in pivot.iterrows():
        cells = [f"<th title='{escape(labels.get(topic_id, ''))}'>Topic {topic_id}</th>"]
        for value in values:
            intensity = 0 if max_value <= 0 else min(float(value) / max_value, 1.0)
            color = f"rgb({255-int(170*intensity)}, {255-int(120*intensity)}, {255-int(60*intensity)})"
            cells.append(f"<td style='background:{color}'>{float(value):.3f}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    header = "".join(f"<th>{escape(str(col))}</th>" for col in pivot.columns)
    legend = "".join(
        f"<li><b>Topic {int(row.topic_id)}</b>: {escape(str(row.topic_words))}</li>"
        for row in summary.sort_values("importance_rank").head(30).itertuples(index=False)
    )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{escape(title)}</title>
<style>body{{font-family:Arial,sans-serif;margin:24px}} table{{border-collapse:collapse}} th,td{{border:1px solid #ddd;padding:6px 8px;font-size:12px}} th{{background:#f5f5f5;position:sticky;top:0}} td{{text-align:right}} .wrap{{overflow:auto;max-height:75vh}}</style>
</head><body><h1>{escape(title)}</h1><p>Cell values are {escape(value_col)}. Hover topic labels for top words.</p><div class="wrap"><table><tr><th>Topic</th>{header}</tr>{''.join(rows)}</table></div><h2>Topic Labels</h2><ol>{legend}</ol></body></html>"""
    path.write_text(html, encoding="utf-8")


def save_bertopic_tables(df: pd.DataFrame, topic_model, topics: list[int], prefix: str, write_legacy: bool) -> None:
    df = df.copy()
    df["topic_id"] = topics
    words = []
    for topic_id in sorted(set(topics)):
        for rank, pair in enumerate(topic_model.get_topic(topic_id) or [], start=1):
            words.append({"topic_id": topic_id, "rank": rank, "word": pair[0], "weight": pair[1]})

    words_df = pd.DataFrame(words)
    chunk_df = df.drop(columns=["embedding_text", "representation_text"], errors="ignore")
    article_summary = (
        df.groupby(["master_id", "topic_id"])
        .size()
        .reset_index(name="chunk_count")
        .sort_values(["master_id", "chunk_count"], ascending=[True, False])
    )
    by_decade = df.groupby(["decade", "topic_id"]).size().reset_index(name="count")
    by_l1 = df.groupby(["L1", "topic_id"]).size().reset_index(name="count")
    summary, by_decade_detailed = build_bertopic_interpretation_tables(df, words_df)

    outputs = {
        "topic_words": words_df,
        "chunk_topics": chunk_df,
        "article_topic_summary": article_summary,
        "topic_by_decade": by_decade,
        "topic_by_decade_detailed": by_decade_detailed,
        "topic_by_L1": by_l1,
        "topic_importance": summary,
        "topic_overall_content_ranked": summary[[
            "importance_rank", "topic_id", "importance_score", "article_coverage", "article_count",
            "topic_size", "topic_size_share", "interpretability", "topic_words",
        ]],
    }
    for suffix, table in outputs.items():
        save_csv(table, TABLES_DIR / f"{prefix}_{suffix}.csv")
        if write_legacy:
            save_csv(table, TABLES_DIR / f"bertopic_{suffix}.csv")
    write_topic_heatmap_html(
        by_decade_detailed,
        summary,
        FIGURES_DIR / f"{prefix}_topic_by_decade_heatmap.html",
        "BERTopic Topic Prevalence by Decade",
    )
    if write_legacy:
        write_topic_heatmap_html(
            by_decade_detailed,
            summary,
            FIGURES_DIR / "bertopic_topic_by_decade_heatmap.html",
            "BERTopic Topic Prevalence by Decade",
        )


def save_bertopic_figures(topic_model, docs: list[str], topics: list[int], years: list[int], prefix: str, write_legacy: bool) -> list[str]:
    warnings = []
    figure_specs = [
        (
            "topic_barchart",
            lambda: topic_model.visualize_barchart(
                top_n_topics=BARCHART_TOP_N_TOPICS,
                n_words=BARCHART_N_WORDS,
            ),
        ),
        ("hierarchy", lambda: topic_model.visualize_hierarchy()),
    ]
    for suffix, builder in figure_specs:
        try:
            fig = builder()
            fig.write_html(FIGURES_DIR / f"{prefix}_{suffix}.html")
            if write_legacy:
                fig.write_html(FIGURES_DIR / f"bertopic_{suffix}.html")
        except Exception as exc:
            warnings.append(f"{suffix}: {exc}")
    try:
        if BARCHART_INCLUDE_ALL_TOPICS:
            all_topics = sorted(topic_id for topic_id in set(topics) if topic_id != -1)
            fig = topic_model.visualize_barchart(
                topics=all_topics,
                n_words=BARCHART_N_WORDS,
            )
            fig.write_html(FIGURES_DIR / f"{prefix}_topic_barchart_all_topics.html")
            if write_legacy:
                fig.write_html(FIGURES_DIR / "bertopic_topic_barchart_all_topics.html")
    except Exception as exc:
        warnings.append(f"topic_barchart_all_topics: {exc}")
    try:
        topics_over_time = topic_model.topics_over_time(docs, topics, years)
        fig = topic_model.visualize_topics_over_time(topics_over_time)
        fig.write_html(FIGURES_DIR / f"{prefix}_topics_over_time.html")
        if write_legacy:
            fig.write_html(FIGURES_DIR / "bertopic_topics_over_time.html")
    except Exception as exc:
        warnings.append(f"topics_over_time: {exc}")
    return warnings


def run_one_variant(config: dict[str, object], embedder, stopwords: set[str], write_legacy: bool) -> BertopicRunResult:
    from bertopic import BERTopic

    name = str(config["name"])
    level = str(config["level"])
    dual_text = bool(config["dual_text"])
    english_only = bool(config["english_only"])
    min_cluster_size = int(config["min_cluster_size"])
    min_samples = int(config["min_samples"])

    df = load_bertopic_input(level, stopwords, dual_text, english_only)
    embedding_docs = df["embedding_text"].astype(str).tolist()
    representation_docs = df["representation_text"].astype(str).tolist()
    embeddings = embedder.encode(embedding_docs, show_progress_bar=True)

    vectorizer_model, ctfidf_model, umap_model, hdbscan_model = make_models(stopwords, min_cluster_size, min_samples)
    topic_model = BERTopic(
        language="multilingual",
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=False,
        verbose=True,
    )
    topics, _ = topic_model.fit_transform(representation_docs, embeddings)
    topics = list(topics)

    if RUN_OUTLIER_REDUCTION and topics.count(-1) > 0:
        try:
            reduced_topics = topic_model.reduce_outliers(representation_docs, topics, strategy="c-tf-idf")
            topic_model.update_topics(
                representation_docs,
                topics=reduced_topics,
                vectorizer_model=vectorizer_model,
                ctfidf_model=ctfidf_model,
            )
            topics = list(reduced_topics)
        except Exception as exc:
            (REPORTS_DIR / f"{name}_outlier_reduction_warning.txt").write_text(str(exc), encoding="utf-8")

    prefix = f"bertopic_{name}"
    save_bertopic_tables(df, topic_model, topics, prefix, write_legacy=write_legacy)
    years = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int).tolist()
    warnings = save_bertopic_figures(topic_model, representation_docs, topics, years, prefix, write_legacy=write_legacy)
    if warnings:
        (REPORTS_DIR / f"{name}_visualization_warning.txt").write_text("\n".join(warnings), encoding="utf-8")

    topic_count = len(set(topics))
    outlier_count = topics.count(-1)
    return BertopicRunResult(
        name=name,
        level=level,
        documents=len(df),
        topic_count=topic_count,
        outlier_count=outlier_count,
        outlier_share=outlier_count / max(len(df), 1),
        output_prefix=prefix,
    )


def report_lines(results: list[BertopicRunResult], stopword_count: int) -> list[str]:
    lines = [
        "# BERTopic Report",
        "",
        "## Pipeline",
        "- Embedding text: cleaned readable article/paragraph/chunk text.",
        "- Topic representation text: filtered text with English, RAS boilerplate, and OCR stopwords removed.",
        f"- Embedding model: `{EMBEDDING_MODEL_NAME}`.",
        f"- Stopwords used: {stopword_count:,}; saved to `data/bertopic_custom_stopwords.txt`.",
        f"- Vectorizer: ngram_range={NGRAM_RANGE}, min_df={MIN_DF}, max_df={MAX_DF}, token_pattern=`{TOKEN_PATTERN}`.",
        f"- c-TF-IDF: bm25_weighting={USE_BM25_WEIGHTING}, reduce_frequent_words={REDUCE_FREQUENT_WORDS}.",
        f"- UMAP: n_neighbors={UMAP_N_NEIGHBORS}, n_components={UMAP_N_COMPONENTS}, min_dist={UMAP_MIN_DIST}.",
        f"- Barchart: top_n_topics={BARCHART_TOP_N_TOPICS}, n_words={BARCHART_N_WORDS}, include_all={BARCHART_INCLUDE_ALL_TOPICS}.",
        f"- Outlier reduction attempted: {RUN_OUTLIER_REDUCTION}.",
        "",
        "## Results",
    ]
    for result in results:
        lines.append(
            f"- {result.name}: level={result.level}, docs={result.documents:,}, "
            f"topics including -1={result.topic_count:,}, outliers={result.outlier_count:,} "
            f"({result.outlier_share:.1%}), prefix=`{result.output_prefix}`"
        )
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "- The legacy `bertopic_*.csv/html` files point to the primary improved variant.",
            "- Use variant-prefixed outputs to compare chunk, paragraph, English-only, and default-like behavior.",
            "- Historical OCR noise can still form topics; inspect representative chunks before assigning labels.",
            "- Multilingual texts are retained unless an English-only variant is selected.",
            "- Topic importance is reported with three retained components: article coverage, topic size, and interpretability.",
            "- Interpretability is operationalized as top-5 c-TF-IDF weight concentration within the top-15 topic words.",
            "- Topic-over-time outputs include detailed decade tables and an HTML heatmap sorted by topic importance.",
        ]
    )
    return lines


def main() -> None:
    ensure_topic_dirs()
    missing = dependency_report(["bertopic", "sentence_transformers", "sklearn"])
    if missing:
        write_placeholder_report(missing)
        print("Missing dependencies: " + ", ".join(missing))
        return

    from sentence_transformers import SentenceTransformer

    stopwords = load_custom_stopwords()
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    configs = VARIANTS if RUN_VARIANT_COMPARISON else [next(v for v in VARIANTS if v["name"] == BERTOPIC_VARIANT)]
    results = []
    for idx, config in enumerate(configs):
        results.append(run_one_variant(config, embedder, stopwords, write_legacy=(idx == 0)))

    summary = pd.DataFrame([result.__dict__ for result in results])
    save_csv(summary, TABLES_DIR / "bertopic_variant_diagnostics.csv")
    write_markdown(REPORTS_DIR / "bertopic_report.md", report_lines(results, len(stopwords)))
    print(f"Wrote improved BERTopic outputs to {TABLES_DIR}")


if __name__ == "__main__":
    main()
