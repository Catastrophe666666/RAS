"""Build article and chunk datasets for RAS topic modeling."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from topic_modeling_utils import (
    DATA_DIR,
    MODEL_MAX_DF,
    MODEL_MAX_FEATURES,
    MODEL_MIN_DF,
    REPORTS_DIR,
    TABLES_DIR,
    build_model_documents,
    ensure_topic_dirs,
    load_model_stopwords,
    normalized_metadata_row,
    read_text,
    save_csv,
    source_paths,
    split_subjects,
    word_count,
    write_markdown,
)


MIN_DOC_WORDS = 100
LONG_DOC_WORDS = 20000
CHUNK_WORDS = 400
CHUNK_OVERLAP_WORDS = 50


def path_from_master(value: object, fallback_dir: Path, filename: str) -> Path:
    if pd.notna(value) and str(value).strip():
        path = Path(str(value))
        if path.exists():
            return path
        candidate = fallback_dir / path.name
        if candidate.exists():
            return candidate
    return fallback_dir / filename


def chunk_words(text: str, chunk_words: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP_WORDS) -> list[str]:
    words = re.findall(r"\S+", text)
    if not words:
        return []
    if len(words) <= chunk_words:
        return [" ".join(words)]
    chunks = []
    step = max(chunk_words - overlap, 1)
    for start in range(0, len(words), step):
        part = words[start : start + chunk_words]
        if len(part) < max(80, chunk_words // 4) and chunks:
            break
        chunks.append(" ".join(part))
    return chunks


def build_articles(master: pd.DataFrame, cleaned_dir: Path, analysis_dir: Path) -> pd.DataFrame:
    rows = []
    for _, row in master.iterrows():
        meta = normalized_metadata_row(row)
        filename = str(meta["filename"]).strip()
        if not filename:
            continue
        cleaned_path = path_from_master(row.get("cleaned_readable_text_path", ""), cleaned_dir, filename)
        analysis_path = path_from_master(row.get("analysis_text_path", ""), analysis_dir, filename)
        cleaned_text = read_text(cleaned_path)
        analysis_text = read_text(analysis_path)
        rows.append(
            {
                "master_id": meta["master_id"],
                "filename": filename,
                "title": meta["title"],
                "author": meta["author"],
                "year": meta["year"],
                "decade": meta["decade"],
                "volume": meta["volume"],
                "issue": meta["issue"],
                "L1": meta["L1"],
                "L2": meta["L2"],
                "language": meta["language"],
                "cleaned_full_text_path": str(cleaned_path),
                "analysis_text_path": str(analysis_path),
                "cleaned_full_text": cleaned_text,
                "analysis_text": analysis_text,
                "word_count": word_count(cleaned_text),
                "analysis_word_count": word_count(analysis_text),
                "character_count": len(cleaned_text),
            }
        )
    return pd.DataFrame(rows)


def build_chunks(articles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, article in articles.iterrows():
        chunks = chunk_words(str(article.get("cleaned_full_text", "")))
        for idx, chunk in enumerate(chunks):
            rows.append(
                {
                    "chunk_id": f"{article.get('master_id')}_{idx:04d}",
                    "master_id": article.get("master_id"),
                    "filename": article.get("filename"),
                    "chunk_index": idx,
                    "chunk_text": chunk,
                    "chunk_word_count": word_count(chunk),
                    "title": article.get("title"),
                    "author": article.get("author"),
                    "year": article.get("year"),
                    "decade": article.get("decade"),
                    "L1": article.get("L1"),
                    "L2": article.get("L2"),
                    "language": article.get("language"),
                    "cleaned_full_text_path": article.get("cleaned_full_text_path"),
                    "analysis_text_path": article.get("analysis_text_path"),
                }
            )
    return pd.DataFrame(rows)


def validation_report(
    articles: pd.DataFrame,
    chunks: pd.DataFrame,
    model_docs: pd.DataFrame,
    model_config: dict[str, object],
    paths: dict[str, Path],
) -> list[str]:
    missing_year = articles["year"].isna() | (articles["year"].astype(str).str.strip() == "")
    missing_title = articles["title"].isna() | (articles["title"].astype(str).str.strip() == "")
    missing_l1 = articles["L1"].isna() | (articles["L1"].astype(str).str.strip() == "")
    missing_l2 = articles["L2"].isna() | (articles["L2"].astype(str).str.strip() == "")
    short_docs = articles[articles["word_count"] < MIN_DOC_WORDS]
    long_docs = articles[articles["word_count"] > LONG_DOC_WORDS]
    non_english = articles[~articles["language"].fillna("").astype(str).str.lower().isin(["", "en", "english"])]

    l1_values = sorted({s for value in articles["L1"].fillna("") for s in split_subjects(value)})[:25]
    language_counts = articles["language"].fillna("unknown").replace("", "unknown").value_counts().to_dict()

    lines = [
        "# Topic Dataset Validation",
        "",
        "## Inputs",
        f"- Master dataset: `{paths['master_dataset']}`",
        f"- Analysis text directory: `{paths['analysis_text_dir']}`",
        f"- Cleaned readable text directory: `{paths['cleaned_text_dir']}`",
        f"- Article cache: `{DATA_DIR / 'topic_modeling_articles.csv'}`",
        f"- Model document cache: `{DATA_DIR / 'topic_modeling_model_documents.csv'}`",
        "",
        "## Counts",
        f"- Article documents: {len(articles):,}",
        f"- Chunk documents: {len(chunks):,}",
        f"- Model documents with usable text: {(model_docs['model_token_count_after_df'] > 0).sum():,}",
        f"- Model documents included after article exclusions: {int(model_docs['model_include'].sum()):,}",
        f"- Model vocabulary size: {model_config['vocabulary_size']:,}",
        f"- Stopwords loaded: {model_config['stopword_count']:,}",
        f"- Document-frequency filter: min_df={model_config['min_df']}, max_df={model_config['max_df']}, max_features={model_config['max_features']}",
        f"- N-gram range: {model_config['ngram_range']}",
        f"- Documents missing year: {int(missing_year.sum()):,}",
        f"- Documents missing title: {int(missing_title.sum()):,}",
        f"- Documents missing L1: {int(missing_l1.sum()):,}",
        f"- Documents missing L2: {int(missing_l2.sum()):,}",
        f"- Very short documents (< {MIN_DOC_WORDS} words): {len(short_docs):,}",
        f"- Unusually long documents (> {LONG_DOC_WORDS} words): {len(long_docs):,}",
        f"- Non-English or mixed-language documents flagged: {len(non_english):,}",
        "",
        "## Language Counts",
    ]
    lines.extend(f"- {lang}: {count}" for lang, count in language_counts.items())
    lines.extend(
        [
            "",
            "## L1 Subjects Seen",
            ", ".join(l1_values) if l1_values else "No L1 subjects found.",
            "",
            "## Source Path Samples",
        ]
    )
    for row in articles.head(8).itertuples(index=False):
        lines.append(f"- `{row.filename}`: readable=`{row.cleaned_full_text_path}`; wordbag=`{row.analysis_text_path}`")
    lines.extend(
        [
            "",
            "## Notes",
            "- `topic_modeling_articles.csv` is a rebuilt cache, not the authoritative source.",
            "- Authoritative text inputs are the readable and wordbag text directories listed above.",
            "- `topic_modeling_model_documents.csv` applies shared stopword removal, tokenization, and document-frequency filtering.",
            "- Long and multilingual texts are retained and flagged rather than removed.",
            "- Chunking uses cleaned readable text, while bag-of-words models should use `analysis_text`.",
        ]
    )
    return lines


def main() -> None:
    ensure_topic_dirs()
    paths = source_paths()
    master = pd.read_csv(paths["master_dataset"])
    articles = build_articles(master, paths["cleaned_text_dir"], paths["analysis_text_dir"])
    chunks = build_chunks(articles)
    model_docs, vocabulary, model_config = build_model_documents(articles)
    chunk_model_docs, chunk_vocabulary, chunk_model_config = build_model_documents(
        chunks,
        text_col="chunk_text",
        min_df=MODEL_MIN_DF,
        max_df=MODEL_MAX_DF,
        max_features=MODEL_MAX_FEATURES,
        min_tokens=40,
        exclude_non_articles=False,
    )
    stopwords = load_model_stopwords()
    save_csv(articles, DATA_DIR / "topic_modeling_articles.csv")
    save_csv(chunks, DATA_DIR / "topic_modeling_chunks.csv")
    save_csv(model_docs, DATA_DIR / "topic_modeling_model_documents.csv")
    save_csv(chunk_model_docs, DATA_DIR / "topic_modeling_chunk_model_documents.csv")
    save_csv(vocabulary, TABLES_DIR / "topic_modeling_model_vocabulary.csv")
    save_csv(chunk_vocabulary, TABLES_DIR / "topic_modeling_chunk_model_vocabulary.csv")
    (DATA_DIR / "stopwords_used_for_modeling.txt").write_text("\n".join(sorted(stopwords)), encoding="utf-8")
    write_markdown(REPORTS_DIR / "topic_dataset_validation.md", validation_report(articles, chunks, model_docs, model_config, paths))
    print(
        f"Wrote {len(articles)} articles, {len(chunks)} chunks, "
        f"{len(model_docs)} article model docs, and {len(chunk_model_docs)} chunk model docs to {DATA_DIR}"
    )


if __name__ == "__main__":
    main()
