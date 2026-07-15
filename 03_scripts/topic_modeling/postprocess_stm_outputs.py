from __future__ import annotations

import html
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
TOPIC_DIR = PROJECT_DIR / "outputs" / "topic_modeling"
TABLES_DIR = TOPIC_DIR / "tables"
FIGURES_DIR = TOPIC_DIR / "figures"
REPORTS_DIR = TOPIC_DIR / "reports"
TOPIC_THRESHOLD = 0.05


def split_subjects(value: object) -> list[str]:
    text = "" if pd.isna(value) else str(value).strip()
    if not text:
        return ["Unknown"]
    parts = [part.strip() for part in text.replace(";", "|").split("|") if part.strip()]
    return parts or ["Unknown"]


def normalize(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        return pd.Series([1.0 if hi > 0 else 0.0] * len(values), index=values.index)
    return (values - lo) / (hi - lo)


def topic_words(words: pd.DataFrame, topic_id: int, word_type: str, n: int) -> str:
    rows = words[
        (words["topic_id"].astype(int) == int(topic_id))
        & (words["word_type"].astype(str) == word_type)
    ].sort_values("rank")
    return ", ".join(rows.head(n)["word"].astype(str).tolist())


def representative_titles(top_docs: pd.DataFrame, n: int = 4) -> list[str]:
    titles: list[str] = []
    for title in top_docs["title"].fillna("").astype(str):
        cleaned = title.strip()
        if cleaned and cleaned not in titles:
            titles.append(cleaned)
        if len(titles) >= n:
            break
    return titles


def write_heatmap_html(heatmap: pd.DataFrame, summary: pd.DataFrame, k: int) -> None:
    top_topics = summary.sort_values("importance_rank").head(30)["topic_id"].tolist()
    data = heatmap[heatmap["topic_id"].isin(top_topics)].copy()
    decades = sorted(data["decade"].astype(str).unique())
    max_value = float(data["mean_topic_proportion"].max()) if len(data) else 0.0
    label_lookup = dict(zip(summary["topic_id"], summary["topic_label"]))
    rows: list[str] = []
    for topic_id in top_topics:
        topic_rows = data[data["topic_id"] == topic_id]
        cells = [
            f"<th title='{html.escape(str(label_lookup.get(topic_id, '')), quote=True)}'>"
            f"Topic {int(topic_id)}</th>"
        ]
        for decade in decades:
            values = topic_rows[topic_rows["decade"].astype(str) == decade]["mean_topic_proportion"]
            value = float(values.iloc[0]) if len(values) else 0.0
            intensity = 0.0 if max_value <= 0 else min(value / max_value, 1.0)
            color = (
                f"rgb({255 - round(170 * intensity)},"
                f"{255 - round(120 * intensity)},"
                f"{255 - round(60 * intensity)})"
            )
            cells.append(f"<td style='background:{color}'>{value:.3f}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    header = "".join(f"<th>{html.escape(str(decade))}</th>" for decade in decades)
    legend = "".join(
        f"<li><b>Topic {int(row.topic_id)}</b>: {html.escape(str(row.topic_label))}</li>"
        for row in summary.sort_values("importance_rank").head(30).itertuples(index=False)
    )
    output = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>STM K {k} Topic Prevalence by Decade</title>
<style>body{{font-family:Arial,sans-serif;margin:24px}} table{{border-collapse:collapse}} th,td{{border:1px solid #ddd;padding:6px 8px;font-size:12px}} th{{background:#f5f5f5;position:sticky;top:0}} td{{text-align:right}}.wrap{{overflow:auto;max-height:75vh}}</style>
</head><body><h1>STM K {k} Topic Prevalence by Decade</h1>
<p>Cell values are mean STM topic proportions by decade. Topic labels combine FREX words, high-probability words, and representative article titles.</p>
<div class="wrap"><table><tr><th>Topic</th>{header}</tr>{''.join(rows)}</table></div>
<h2>Topic Labels</h2><ol>{legend}</ol></body></html>"""
    (FIGURES_DIR / f"stm_K{k}_topic_by_decade_heatmap.html").write_text(output, encoding="utf-8")


def postprocess_k(k: int) -> None:
    words_path = TABLES_DIR / f"stm_K{k}_topic_words.csv"
    docs_path = TABLES_DIR / f"stm_K{k}_document_topics.csv"
    if not words_path.exists() or not docs_path.exists():
        return
    words = pd.read_csv(words_path)
    docs = pd.read_csv(docs_path)
    topic_cols = [col for col in docs.columns if col.startswith("topic_")]

    representative_rows = []
    l1_rows = []
    summary_rows = []
    for topic_col in topic_cols:
        topic_id = int(topic_col.replace("topic_", ""))
        top_docs = docs.sort_values(topic_col, ascending=False).head(12)
        frex = topic_words(words, topic_id, "frex", 10)
        prob = topic_words(words, topic_id, "prob", 10)
        title_bits = representative_titles(top_docs, 4)
        topic_label = f"FREX: {frex} | Prob: {prob} | Representative: {'; '.join(title_bits)}"

        for rank, row in enumerate(top_docs.itertuples(index=False), start=1):
            representative_rows.append(
                {
                    "K": k,
                    "topic_id": topic_id,
                    "rank": rank,
                    "theta": float(getattr(row, topic_col)),
                    "master_id": getattr(row, "master_id", ""),
                    "title": getattr(row, "title", ""),
                    "author": getattr(row, "author", ""),
                    "year": getattr(row, "year", ""),
                    "decade": getattr(row, "decade", ""),
                    "L1": getattr(row, "L1", ""),
                    "topic_label": topic_label,
                }
            )

        exploded = []
        for idx, row in docs.iterrows():
            for subject in split_subjects(row.get("L1", "")):
                exploded.append((subject, float(row[topic_col])))
        exploded_df = pd.DataFrame(exploded, columns=["L1", "theta"])
        grouped = exploded_df.groupby("L1").agg(
            mean_topic_proportion=("theta", "mean"),
            article_count=("theta", lambda values: int((values >= TOPIC_THRESHOLD).sum())),
            article_coverage=("theta", lambda values: float((values >= TOPIC_THRESHOLD).mean())),
        )
        grouped = grouped.sort_values("mean_topic_proportion", ascending=False).reset_index()
        for row in grouped.itertuples(index=False):
            l1_rows.append(
                {
                    "K": k,
                    "topic_id": topic_id,
                    "L1": row.L1,
                    "mean_topic_proportion": row.mean_topic_proportion,
                    "article_count": row.article_count,
                    "article_coverage": row.article_coverage,
                    "topic_label": topic_label,
                }
            )
        summary_rows.append(
            {
                "K": k,
                "topic_id": topic_id,
                "topic_label": topic_label,
                "frex_words": frex,
                "prob_words": prob,
                "representative_titles": "; ".join(title_bits),
            }
        )

    representatives = pd.DataFrame(representative_rows)
    by_l1 = pd.DataFrame(l1_rows)
    labels = pd.DataFrame(summary_rows)
    representatives.to_csv(TABLES_DIR / f"stm_K{k}_representative_articles.csv", index=False, encoding="utf-8-sig")
    by_l1.to_csv(TABLES_DIR / f"stm_K{k}_topic_by_L1_long.csv", index=False, encoding="utf-8-sig")
    labels.to_csv(TABLES_DIR / f"stm_K{k}_topic_labels_improved.csv", index=False, encoding="utf-8-sig")

    importance_path = TABLES_DIR / f"stm_K{k}_topic_importance.csv"
    by_decade_path = TABLES_DIR / f"stm_K{k}_topic_by_decade_long.csv"
    if importance_path.exists():
        importance = pd.read_csv(importance_path)
        importance = importance.drop(columns=["topic_label"], errors="ignore").merge(labels[["topic_id", "topic_label"]], on="topic_id", how="left")
        importance.to_csv(importance_path, index=False, encoding="utf-8-sig")
        content_cols = [
            col for col in [
                "importance_rank",
                "topic_id",
                "importance_score",
                "article_coverage",
                "article_count",
                "topic_size",
                "topic_size_share",
                "interpretability",
                "semantic_coherence",
                "exclusivity",
                "topic_label",
                "topic_words",
                "prob_words",
            ]
            if col in importance.columns
        ]
        importance[content_cols].to_csv(TABLES_DIR / f"stm_K{k}_topic_overall_content_ranked.csv", index=False, encoding="utf-8-sig")
        if by_decade_path.exists():
            by_decade = pd.read_csv(by_decade_path)
            by_decade = by_decade.drop(columns=["topic_label"], errors="ignore").merge(labels[["topic_id", "topic_label"]], on="topic_id", how="left")
            by_decade.to_csv(by_decade_path, index=False, encoding="utf-8-sig")
            write_heatmap_html(by_decade, importance, k)


def write_k_selection_report() -> None:
    diag_path = TABLES_DIR / "stm_diagnostics.csv"
    if not diag_path.exists():
        return
    diag = pd.read_csv(diag_path)
    diag["coherence_norm"] = normalize(diag["semantic_coherence_mean"])
    diag["exclusivity_norm"] = normalize(diag["exclusivity_mean"])
    diag["balanced_score"] = (diag["coherence_norm"] + diag["exclusivity_norm"]) / 2
    diag["coherence_weighted_score"] = 0.6 * diag["coherence_norm"] + 0.4 * diag["exclusivity_norm"]
    diag = diag.sort_values("K")
    best_balanced = int(diag.sort_values("balanced_score", ascending=False).iloc[0]["K"])
    best_coherence_weighted = int(diag.sort_values("coherence_weighted_score", ascending=False).iloc[0]["K"])
    diag.to_csv(TABLES_DIR / "stm_k_selection_scores.csv", index=False, encoding="utf-8-sig")
    score_cols = ["K", "semantic_coherence_mean", "exclusivity_mean", "coherence_norm", "exclusivity_norm", "balanced_score", "coherence_weighted_score"]
    markdown_rows = [
        "| " + " | ".join(score_cols) + " |",
        "| " + " | ".join(["---"] * len(score_cols)) + " |",
    ]
    for row in diag[score_cols].itertuples(index=False):
        markdown_rows.append("| " + " | ".join(str(round(value, 4)) if isinstance(value, float) else str(value) for value in row) + " |")
    lines = [
        "# STM K Selection",
        "",
        "Semantic coherence is better when it is higher, meaning less negative here. Exclusivity is better when higher.",
        "",
        f"- Best balanced K by normalized coherence and exclusivity: K={best_balanced}",
        f"- Best coherence-weighted K: K={best_coherence_weighted}",
        "- Recommendation for the main reported model: K=15, because it is the elbow where exclusivity improves substantially over K=10, while gains after K=15 are small and coherence continues to deteriorate.",
        "- Use K=10 as a simpler robustness check if the priority is maximum semantic coherence and fewer topics.",
        "",
        "## Scores",
        "",
        "\n".join(markdown_rows),
    ]
    (REPORTS_DIR / "stm_k_selection_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    for k in [10, 15, 20, 25, 30]:
        postprocess_k(k)
    write_k_selection_report()


if __name__ == "__main__":
    main()
