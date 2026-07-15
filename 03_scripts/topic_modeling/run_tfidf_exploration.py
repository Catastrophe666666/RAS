"""TF-IDF diagnostics plus a lightweight LDA probe for RAS topic modeling."""

from __future__ import annotations

import math
import re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from topic_modeling_utils import (
    FIGURES_DIR,
    MODEL_MAX_DF,
    MODEL_MAX_FEATURES,
    MODEL_MIN_DF,
    RANDOM_SEED,
    REPORTS_DIR,
    TABLES_DIR,
    ensure_topic_dirs,
    load_chunks,
    load_model_documents,
    save_csv,
    write_markdown,
)


TOP_N_OVERALL = 100
TOP_N_GROUP = 25
RUN_LDA_PROBE = False
LDA_K_VALUES = [5, 8, 10]
LDA_TOP_N_WORDS = 15
LDA_ITERATIONS = 8
LDA_MAX_TOKENS_PER_DOC = 50
CHUNK_LDA_MAX_DOCS = 800
LDA_ALPHA = 0.10
LDA_BETA = 0.01
WORDCLOUD_WARNING_EMITTED = False


def split_tokenize(text: str) -> list[str]:
    return str(text).split()


def draw_bar_chart(df: pd.DataFrame, path: Path, title: str, label_col: str = "term", value_col: str = "tfidf_score") -> None:
    top = df.head(25).copy()
    width, row_h, margin = 1200, 32, 40
    height = margin * 2 + row_h * (len(top) + 1)
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.text((margin, 16), title, fill=(20, 20, 20), font=font)
    max_value = float(top[value_col].max()) if len(top) else 1.0
    bar_x = 300
    bar_w = width - bar_x - margin
    for i, row in enumerate(top.itertuples(index=False), start=0):
        y = margin + 28 + i * row_h
        label = str(getattr(row, label_col))[:36]
        value = float(getattr(row, value_col))
        draw.text((margin, y + 6), label, fill=(40, 40, 40), font=font)
        fill_w = int(bar_w * value / max_value) if max_value else 0
        draw.rectangle([bar_x, y, bar_x + fill_w, y + 20], fill=(70, 118, 170))
        draw.text((bar_x + fill_w + 8, y + 4), f"{value:.3f}", fill=(40, 40, 40), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def draw_group_chart(df: pd.DataFrame, group_col: str, path: Path, title: str) -> None:
    if df.empty:
        draw_bar_chart(pd.DataFrame(columns=["term", "tfidf_score"]), path, title)
        return
    rows = []
    for group, group_df in df.groupby(group_col):
        terms = ", ".join(group_df.sort_values("rank").head(8)["term"].astype(str))
        rows.append({"term": str(group), "tfidf_score": float(group_df["tfidf_score"].head(8).sum()), "terms": terms})
    chart_df = pd.DataFrame(rows).sort_values("tfidf_score", ascending=False).head(25)
    draw_bar_chart(chart_df, path, title)


def draw_word_cloud(df: pd.DataFrame, path: Path, title: str, label_col: str = "term", value_col: str = "tfidf_score") -> None:
    global WORDCLOUD_WARNING_EMITTED
    try:
        from wordcloud import WordCloud
    except ImportError:
        warning = FIGURES_DIR / "tfidf_wordcloud_package_missing.txt"
        if not WORDCLOUD_WARNING_EMITTED:
            warning.write_text(
                "TF-IDF word clouds were not generated because the Python package "
                "`wordcloud` is not installed in this environment.\n"
                "Install it with: python -m pip install wordcloud matplotlib\n",
                encoding="utf-8",
            )
            print("Warning: wordcloud package is not installed; skipped TF-IDF word clouds.")
            WORDCLOUD_WARNING_EMITTED = True
        return
    frequencies = {
        str(getattr(row, label_col)).replace("_", " "): float(getattr(row, value_col))
        for row in df.head(120).itertuples(index=False)
        if float(getattr(row, value_col)) > 0
    }
    if not frequencies:
        return
    wc = WordCloud(
        width=1400,
        height=900,
        background_color="white",
        colormap="tab20",
        collocations=False,
        prefer_horizontal=0.9,
        random_state=RANDOM_SEED,
        max_words=120,
        margin=4,
    ).generate_from_frequencies(frequencies)
    path.parent.mkdir(parents=True, exist_ok=True)
    wc.to_file(str(path))


def draw_group_word_clouds(df: pd.DataFrame, group_col: str, prefix: str, max_groups: int = 20) -> None:
    if df.empty or group_col not in df.columns:
        return
    group_scores = (
        df.groupby(group_col)["tfidf_score"]
        .sum()
        .sort_values(ascending=False)
        .head(max_groups)
    )
    for group_value in group_scores.index:
        group_df = df[df[group_col] == group_value].sort_values("rank")
        safe = re_safe_filename(str(group_value))
        draw_word_cloud(group_df, FIGURES_DIR / f"{prefix}_{safe}_wordcloud.png", f"TF-IDF Word Cloud: {group_value}")
        draw_bar_chart(group_df, FIGURES_DIR / f"{prefix}_{safe}_bar.png", f"Top TF-IDF Terms: {group_value}")


def re_safe_filename(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")[:80] or "unknown"


def draw_lda_diagnostics(diag: pd.DataFrame, path: Path) -> None:
    width, height, margin = 1000, 520, 70
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.text((margin, 24), "LDA Probe Diagnostics", fill=(20, 20, 20), font=font)
    if diag.empty:
        img.save(path)
        return
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    xs = diag["K"].astype(float).to_numpy()
    perp = diag["perplexity"].astype(float).to_numpy()
    conc = diag["document_topic_concentration"].astype(float).to_numpy()
    x_min, x_max = xs.min(), xs.max()
    p_min, p_max = perp.min(), perp.max()

    def x_coord(x: float) -> int:
        return int(margin + (x - x_min) / max(x_max - x_min, 1) * plot_w)

    def y_coord(value: float, min_value: float, max_value: float) -> int:
        return int(height - margin - (value - min_value) / max(max_value - min_value, 1e-9) * plot_h)

    draw.line([margin, height - margin, width - margin, height - margin], fill=(80, 80, 80))
    draw.line([margin, margin, margin, height - margin], fill=(80, 80, 80))
    points = [(x_coord(x), y_coord(y, p_min, p_max)) for x, y in zip(xs, perp)]
    if len(points) > 1:
        draw.line(points, fill=(31, 119, 180), width=3)
    for x, y, k in zip([p[0] for p in points], [p[1] for p in points], xs):
        draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(31, 119, 180))
        draw.text((x - 8, height - margin + 10), str(int(k)), fill=(40, 40, 40), font=font)
    draw.text((margin, height - 34), "K", fill=(40, 40, 40), font=font)
    draw.text((margin + 5, margin - 20), "perplexity, lower is better", fill=(31, 119, 180), font=font)
    for _, row in diag.iterrows():
        x = x_coord(float(row["K"]))
        y = int(height - margin - float(row["document_topic_concentration"]) * plot_h)
        draw.rectangle([x - 5, y, x + 5, height - margin], fill=(180, 210, 140))
    draw.text((width - 280, margin - 20), "green bars: topic concentration", fill=(70, 110, 50), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def draw_topic_word_heatmap(topic_words: pd.DataFrame, path: Path, k: int) -> None:
    top = topic_words[(topic_words["K"] == k) & (topic_words["rank"] <= 8)].copy()
    width, row_h, margin = 1400, 34, 40
    height = margin * 2 + row_h * (k + 1)
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.text((margin, 16), f"LDA K={k} Top Words", fill=(20, 20, 20), font=font)
    for topic_id, group in top.groupby("topic_id"):
        y = margin + 28 + int(topic_id) * row_h
        words = "   ".join(group.sort_values("rank")["word"].astype(str).tolist())
        draw.text((margin, y + 6), f"topic_{topic_id}", fill=(40, 40, 40), font=font)
        draw.text((160, y + 6), words[:170], fill=(40, 40, 40), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def count_documents(texts: list[str]) -> tuple[list[Counter], list[str], Counter, Counter]:
    doc_counts = []
    raw_counts = Counter()
    document_counts = Counter()
    for text in texts:
        counts = Counter(split_tokenize(text))
        doc_counts.append(counts)
        raw_counts.update(counts)
        document_counts.update(counts.keys())
    terms = sorted(raw_counts)
    return doc_counts, terms, raw_counts, document_counts


def tfidf_overall_from_counts(
    doc_counts: list[Counter],
    terms: list[str],
    raw_counts: Counter,
    document_counts: Counter,
    top_n: int,
) -> pd.DataFrame:
    n_docs = max(len(doc_counts), 1)
    allowed_terms = set(terms)
    scores = Counter()
    for counts in doc_counts:
        total = sum(counts.values()) or 1
        for term, count in counts.items():
            if term not in allowed_terms:
                continue
            idf = math.log((1 + n_docs) / (1 + document_counts[term])) + 1
            scores[term] += (count / total) * idf
    top_terms = scores.most_common(top_n)
    return pd.DataFrame(
        [
            {
                "rank": rank,
                "term": term,
                "tfidf_score": float(score),
                "raw_count": int(raw_counts[term]),
                "document_count": int(document_counts[term]),
                "document_frequency": float(document_counts[term] / n_docs),
            }
            for rank, (term, score) in enumerate(top_terms, start=1)
        ]
    )


def split_group_values(value: object) -> list[str]:
    text = "" if pd.isna(value) else str(value).strip()
    if not text:
        return ["Unknown"]
    parts = [part.strip() for part in re.split(r"\s*\|\s*|\s*;\s*", text) if part.strip()]
    return parts or ["Unknown"]


def grouped_tfidf_from_counts(articles: pd.DataFrame, doc_counts: list[Counter], group_col: str, top_n: int) -> pd.DataFrame:
    rows = []
    if group_col not in articles.columns:
        return pd.DataFrame()
    group_to_indexes: dict[str, list[int]] = {}
    for idx, value in articles[group_col].items():
        values = split_group_values(value) if group_col in {"L1", "L2"} else [str(value).strip() or "Unknown"]
        for group_value in values:
            group_to_indexes.setdefault(group_value, []).append(idx)
    for group_value, idx_values in group_to_indexes.items():
        selected = [doc_counts[idx] for idx in idx_values]
        raw_counts = Counter()
        document_counts = Counter()
        for counts in selected:
            raw_counts.update(counts)
            document_counts.update(counts.keys())
        terms = sorted(raw_counts)
        table = tfidf_overall_from_counts(selected, terms, raw_counts, document_counts, top_n)
        for row in table.itertuples(index=False):
            rows.append(
                {
                    group_col: group_value,
                    "rank": row.rank,
                    "term": row.term,
                    "tfidf_score": row.tfidf_score,
                    "raw_count": row.raw_count,
                    "document_count": row.document_count,
                    "document_frequency": row.document_frequency,
                }
            )
    return pd.DataFrame(rows)


def sampled_doc_word_ids(texts: list[str], term_to_id: dict[str, int]) -> list[list[int]]:
    rng = np.random.default_rng(RANDOM_SEED)
    docs = []
    for text in texts:
        ids = [term_to_id[token] for token in split_tokenize(text) if token in term_to_id]
        if len(ids) > LDA_MAX_TOKENS_PER_DOC:
            ids = rng.choice(ids, size=LDA_MAX_TOKENS_PER_DOC, replace=False).tolist()
        docs.append(ids)
    return docs


def run_numpy_lda(doc_word_ids: list[list[int]], vocab_size: int, k: int) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(RANDOM_SEED + k)
    n_docs = len(doc_word_ids)
    doc_topic = np.zeros((n_docs, k), dtype=np.int32)
    topic_word = np.zeros((k, vocab_size), dtype=np.int32)
    topic_totals = np.zeros(k, dtype=np.int32)
    assignments = []
    for d, words in enumerate(doc_word_ids):
        doc_assignments = rng.integers(0, k, size=len(words), endpoint=False).tolist()
        assignments.append(doc_assignments)
        for word_id, topic in zip(words, doc_assignments):
            doc_topic[d, topic] += 1
            topic_word[topic, word_id] += 1
            topic_totals[topic] += 1
    beta_sum = LDA_BETA * vocab_size
    for _ in range(LDA_ITERATIONS):
        for d, words in enumerate(doc_word_ids):
            for i, word_id in enumerate(words):
                old_topic = assignments[d][i]
                doc_topic[d, old_topic] -= 1
                topic_word[old_topic, word_id] -= 1
                topic_totals[old_topic] -= 1
                probs = (doc_topic[d] + LDA_ALPHA) * (topic_word[:, word_id] + LDA_BETA) / (topic_totals + beta_sum)
                probs = probs / probs.sum()
                new_topic = int(rng.choice(k, p=probs))
                assignments[d][i] = new_topic
                doc_topic[d, new_topic] += 1
                topic_word[new_topic, word_id] += 1
                topic_totals[new_topic] += 1
    theta = (doc_topic + LDA_ALPHA) / (doc_topic.sum(axis=1, keepdims=True) + LDA_ALPHA * k)
    phi = (topic_word + LDA_BETA) / (topic_totals[:, None] + beta_sum)
    log_likelihood = 0.0
    token_count = 0
    for d, words in enumerate(doc_word_ids):
        for word_id in words:
            prob = float(np.dot(theta[d], phi[:, word_id]))
            log_likelihood += math.log(max(prob, 1e-12))
            token_count += 1
    pseudo_perplexity = math.exp(-log_likelihood / max(token_count, 1))
    return theta, phi, pseudo_perplexity


def load_chunk_model_documents() -> pd.DataFrame:
    path = TABLES_DIR.parent / "data" / "topic_modeling_chunk_model_documents.csv"
    if path.exists():
        return pd.read_csv(path)
    chunks = load_chunks()
    from topic_modeling_utils import build_model_documents

    docs, _, _ = build_model_documents(chunks, text_col="chunk_text", min_tokens=40, exclude_non_articles=False)
    return docs


def lda_probe(
    articles: pd.DataFrame,
    texts: list[str],
    terms: list[str],
    raw_counts: Counter,
    model_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, pd.DataFrame], dict[int, dict[str, object]]]:
    terms = sorted(terms, key=lambda term: (-raw_counts[term], term))[:MODEL_MAX_FEATURES]
    term_to_id = {term: idx for idx, term in enumerate(terms)}
    doc_word_ids = sampled_doc_word_ids(texts, term_to_id)
    diagnostics = []
    word_rows = []
    doc_topic_tables = {}
    pyldavis_inputs = {}
    for k in LDA_K_VALUES:
        doc_topic, topic_word, perplexity = run_numpy_lda(doc_word_ids, len(terms), k)
        topic_mass = doc_topic.sum(axis=0)
        diagnostics.append(
            {
                "corpus_level": model_label,
                "K": k,
                "perplexity": float(perplexity),
                "log_likelihood_bound": np.nan,
                "document_topic_concentration": float(doc_topic.max(axis=1).mean()),
                "near_empty_topics": int((topic_mass < 0.01).sum()),
                "iterations": LDA_ITERATIONS,
                "max_tokens_per_doc": LDA_MAX_TOKENS_PER_DOC,
            }
        )
        for topic_idx, weights in enumerate(topic_word):
            order = np.argsort(weights)[::-1][:LDA_TOP_N_WORDS]
            for rank, idx in enumerate(order, start=1):
                word_rows.append(
                    {
                        "model": "lda_probe",
                        "corpus_level": model_label,
                        "K": k,
                        "topic_id": topic_idx,
                        "rank": rank,
                        "word": terms[idx],
                        "weight": float(weights[idx]),
                    }
                )
        topic_cols = {f"topic_{i}": doc_topic[:, i] for i in range(k)}
        metadata_cols = [
            "chunk_id", "chunk_index", "master_id", "filename", "title", "author", "year", "decade", "L1", "L2", "language",
            "cleaned_full_text_path", "analysis_text_path", "model_source_column",
            "model_exclusion_reason", "model_unigram_count_before_df",
            "model_token_count_before_df", "model_token_count_after_df",
        ]
        metadata_cols = [col for col in metadata_cols if col in articles.columns]
        doc_table = pd.concat(
            [
                articles[metadata_cols].reset_index(drop=True),
                pd.DataFrame(topic_cols),
            ],
            axis=1,
        )
        doc_table["dominant_topic"] = doc_topic.argmax(axis=1)
        doc_table["dominant_topic_weight"] = doc_topic.max(axis=1)
        doc_topic_tables[k] = doc_table
        pyldavis_inputs[k] = {
            "topic_term_dists": topic_word,
            "doc_topic_dists": doc_topic,
            "doc_lengths": np.array([len(words) for words in doc_word_ids], dtype=int),
            "vocab": terms,
            "term_frequency": np.array([raw_counts[term] for term in terms], dtype=float),
        }
    return pd.DataFrame(diagnostics), pd.DataFrame(word_rows), doc_topic_tables, pyldavis_inputs


def save_pyldavis_inputs(inputs: dict[str, object], prefix: str) -> None:
    save_csv(pd.DataFrame(inputs["topic_term_dists"], columns=inputs["vocab"]), TABLES_DIR / f"{prefix}_pyldavis_topic_term_dists.csv")
    save_csv(pd.DataFrame(inputs["doc_topic_dists"], columns=[f"topic_{i}" for i in range(inputs["doc_topic_dists"].shape[1])]), TABLES_DIR / f"{prefix}_pyldavis_doc_topic_dists.csv")
    save_csv(pd.DataFrame({"doc_length": inputs["doc_lengths"]}), TABLES_DIR / f"{prefix}_pyldavis_doc_lengths.csv")
    save_csv(pd.DataFrame({"term": inputs["vocab"], "term_frequency": inputs["term_frequency"]}), TABLES_DIR / f"{prefix}_pyldavis_term_frequency.csv")


def report_lines(
    articles: pd.DataFrame,
    overall: pd.DataFrame,
    lda_diag: pd.DataFrame,
    best_k: int | None,
    chunk_diag: pd.DataFrame,
    chunk_best_k: int | None,
) -> list[str]:
    source_paths = articles[["filename", "cleaned_full_text_path", "analysis_text_path"]].head(8)
    lines = [
        "# TF-IDF Exploration and LDA Probe Report",
        "",
        "## Corpus",
        f"- Documents analyzed: {len(articles):,}",
        f"- Documents excluded before article-level TF-IDF/LDA: {int((~articles.get('model_include', pd.Series([True] * len(articles))).astype(bool)).sum()):,}",
        f"- Documents with post-filter model tokens: {(articles['model_token_count_after_df'] > 0).sum():,}",
        f"- Overall TF-IDF terms saved: {len(overall):,}",
        f"- Tokenizer: Latin alphabet tokens, lowercased, length >= 3, plus bigrams/trigrams encoded with underscores.",
        f"- Stopword removal: RAS-first `hit_stopwords.txt`, RAS domain stopwords, and optional English stopwords when available.",
        f"- Document-frequency filter: min_df={MODEL_MIN_DF}, max_df={MODEL_MAX_DF}, max_features={MODEL_MAX_FEATURES}.",
        "",
        "## Source Path Samples",
    ]
    for row in source_paths.itertuples(index=False):
        lines.append(f"- `{row.filename}`: readable=`{row.cleaned_full_text_path}`; wordbag=`{row.analysis_text_path}`")
    lines.extend(
        [
            "",
            "## LDA Probe",
        ]
    )
    if lda_diag.empty:
        lines.append("- LDA probe was not run.")
    else:
        for row in lda_diag.sort_values("K").itertuples(index=False):
            lines.append(
                f"- K={row.K}: pseudo-perplexity={row.perplexity:.2f}, concentration={row.document_topic_concentration:.3f}, near-empty={row.near_empty_topics}"
            )
        lines.append(f"- Candidate K by lowest pseudo-perplexity with no near-empty topics where possible: {best_k}")
        lines.append(f"- LDA probe engine: NumPy collapsed Gibbs sampler, iterations={LDA_ITERATIONS}, max_tokens_per_doc={LDA_MAX_TOKENS_PER_DOC}.")
    lines.extend(["", "## Chunk-Level LDA Probe"])
    if chunk_diag.empty:
        lines.append("- Chunk-level LDA probe was not run.")
    else:
        for row in chunk_diag.sort_values("K").itertuples(index=False):
            lines.append(
                f"- K={row.K}: pseudo-perplexity={row.perplexity:.2f}, concentration={row.document_topic_concentration:.3f}, near-empty={row.near_empty_topics}"
            )
        lines.append(f"- Chunk-level candidate K: {chunk_best_k}")
    lines.extend(
        [
            "",
            "TF-IDF is used as vocabulary diagnostics and stratified signal review. LDA uses count-style token lists from the shared model documents, not TF-IDF weights.",
        ]
    )
    return lines


def main() -> None:
    ensure_topic_dirs()
    all_articles = load_model_documents()
    include = all_articles["model_include"].fillna(True).astype(bool) if "model_include" in all_articles else True
    articles = all_articles[include & (all_articles["model_text"].fillna("").astype(str).str.len() > 0)].reset_index(drop=True)

    texts = articles["model_text"].fillna("").astype(str).tolist()
    doc_counts, terms, raw_counts, document_counts = count_documents(texts)

    print("Computing TF-IDF tables...")
    overall = tfidf_overall_from_counts(doc_counts, terms, raw_counts, document_counts, TOP_N_OVERALL)
    phrase_terms = [term for term in terms if "_" in str(term)]
    overall_phrases = tfidf_overall_from_counts(doc_counts, phrase_terms, raw_counts, document_counts, TOP_N_OVERALL)
    by_decade = grouped_tfidf_from_counts(articles, doc_counts, "decade", TOP_N_GROUP)
    by_l1 = grouped_tfidf_from_counts(articles, doc_counts, "L1", TOP_N_GROUP)
    by_l2 = grouped_tfidf_from_counts(articles, doc_counts, "L2", TOP_N_GROUP)

    save_csv(overall, TABLES_DIR / "tfidf_top_terms_overall.csv")
    save_csv(overall_phrases, TABLES_DIR / "tfidf_top_phrases_overall.csv")
    save_csv(by_decade, TABLES_DIR / "tfidf_top_terms_by_decade.csv")
    save_csv(by_l1, TABLES_DIR / "tfidf_top_terms_by_L1.csv")
    save_csv(by_l2, TABLES_DIR / "tfidf_top_terms_by_L2.csv")
    save_csv(
        all_articles[[
            col for col in [
                "master_id", "filename", "title", "year", "decade", "L1", "L2",
                "model_include", "model_exclusion_reason", "model_unigram_count_before_df",
                "model_token_count_before_df", "model_token_count_after_df",
                "cleaned_full_text_path", "analysis_text_path",
            ]
            if col in all_articles.columns
        ]],
        TABLES_DIR / "tfidf_article_model_exclusions.csv",
    )

    draw_bar_chart(overall, FIGURES_DIR / "tfidf_top_terms_overall.png", "Top TF-IDF Terms Overall")
    draw_bar_chart(overall_phrases, FIGURES_DIR / "tfidf_top_phrases_overall.png", "Top TF-IDF Phrases Overall")
    draw_group_chart(by_decade, "decade", FIGURES_DIR / "tfidf_top_terms_by_decade.png", "TF-IDF Signal by Decade")
    draw_group_chart(by_l1, "L1", FIGURES_DIR / "tfidf_top_terms_by_L1.png", "TF-IDF Signal by L1 Subject")
    draw_word_cloud(overall, FIGURES_DIR / "tfidf_top_terms_overall_wordcloud.png", "TF-IDF Word Cloud Overall")
    draw_group_word_clouds(by_decade, "decade", "tfidf_by_decade")
    draw_group_word_clouds(by_l1, "L1", "tfidf_by_L1")

    lda_diag = pd.DataFrame()
    chunk_diag = pd.DataFrame()
    best_k = None
    chunk_best_k = None
    if RUN_LDA_PROBE:
        print("Running article-level LDA probe...")
        lda_diag, lda_words, lda_docs, pyldavis_inputs = lda_probe(articles, texts, terms, raw_counts, "article")
        save_csv(lda_diag, TABLES_DIR / "tfidf_lda_probe_diagnostics.csv")
        save_csv(lda_words, TABLES_DIR / "tfidf_lda_probe_topic_words.csv")
        candidate_pool = lda_diag[lda_diag["near_empty_topics"] == 0] if not lda_diag.empty else lda_diag
        if candidate_pool.empty:
            candidate_pool = lda_diag
        best_k = int(candidate_pool.sort_values(["perplexity", "document_topic_concentration"], ascending=[True, False]).iloc[0]["K"]) if not candidate_pool.empty else None
        for k, docs in lda_docs.items():
            suffix = "_candidate" if k == best_k else ""
            save_csv(docs, TABLES_DIR / f"tfidf_lda_probe_K{k}_document_topics{suffix}.csv")
        draw_lda_diagnostics(lda_diag, FIGURES_DIR / "tfidf_lda_probe_diagnostics.png")
        if best_k is not None:
            draw_topic_word_heatmap(lda_words, FIGURES_DIR / f"tfidf_lda_probe_K{best_k}_topic_words.png", best_k)
            save_pyldavis_inputs(pyldavis_inputs[best_k], f"tfidf_lda_probe_article_K{best_k}")

    write_markdown(REPORTS_DIR / "tfidf_exploration_report.md", report_lines(all_articles, overall, lda_diag, best_k, chunk_diag, chunk_best_k))
    print(f"Wrote TF-IDF outputs to {TABLES_DIR}")


if __name__ == "__main__":
    main()
