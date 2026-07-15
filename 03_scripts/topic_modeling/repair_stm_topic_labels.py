from __future__ import annotations

import html
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
TOPIC_DIR = PROJECT_DIR / "outputs" / "topic_modeling"
TABLES_DIR = TOPIC_DIR / "tables"
FIGURES_DIR = TOPIC_DIR / "figures"


def topic_label(words: pd.DataFrame, topic_id: int, word_type: str = "frex", n: int = 12) -> str:
    selected = words[
        (words["topic_id"].astype(int) == int(topic_id))
        & (words["word_type"].astype(str) == word_type)
    ].sort_values("rank")
    return ", ".join(selected.head(n)["word"].astype(str).tolist())


def rewrite_k(k: int) -> None:
    words_path = TABLES_DIR / f"stm_K{k}_topic_words.csv"
    importance_path = TABLES_DIR / f"stm_K{k}_topic_importance.csv"
    content_path = TABLES_DIR / f"stm_K{k}_topic_overall_content_ranked.csv"
    long_path = TABLES_DIR / f"stm_K{k}_topic_by_decade_long.csv"
    html_path = FIGURES_DIR / f"stm_K{k}_topic_by_decade_heatmap.html"

    words = pd.read_csv(words_path)
    importance = pd.read_csv(importance_path)
    by_decade = pd.read_csv(long_path)

    importance["topic_words"] = importance["topic_id"].map(lambda topic_id: topic_label(words, topic_id, "frex", 12))
    importance["prob_words"] = importance["topic_id"].map(lambda topic_id: topic_label(words, topic_id, "prob", 12))
    importance.to_csv(importance_path, index=False, encoding="utf-8-sig")

    content_cols = [
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
        "topic_words",
        "prob_words",
    ]
    importance[content_cols].to_csv(content_path, index=False, encoding="utf-8-sig")

    by_decade["topic_words"] = by_decade["topic_id"].map(lambda topic_id: topic_label(words, topic_id, "frex", 8))
    by_decade.to_csv(long_path, index=False, encoding="utf-8-sig")

    top_topics = importance.sort_values("importance_rank").head(30)["topic_id"].tolist()
    heatmap = by_decade[by_decade["topic_id"].isin(top_topics)].copy()
    decades = sorted(heatmap["decade"].astype(str).unique())
    max_value = float(heatmap["mean_topic_proportion"].max()) if len(heatmap) else 0.0
    label_lookup = dict(zip(importance["topic_id"], importance["topic_words"]))

    rows: list[str] = []
    for topic_id in top_topics:
        topic_rows = heatmap[heatmap["topic_id"] == topic_id]
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
        f"<li><b>Topic {int(row.topic_id)}</b>: {html.escape(str(row.topic_words))}</li>"
        for row in importance.sort_values("importance_rank").head(30).itertuples(index=False)
    )
    output = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>STM K {k} Topic Prevalence by Decade</title>
<style>body{{font-family:Arial,sans-serif;margin:24px}} table{{border-collapse:collapse}} th,td{{border:1px solid #ddd;padding:6px 8px;font-size:12px}} th{{background:#f5f5f5;position:sticky;top:0}} td{{text-align:right}}.wrap{{overflow:auto;max-height:75vh}}</style>
</head><body><h1>STM K {k} Topic Prevalence by Decade</h1>
<p>Cell values are mean STM topic proportions by decade. Hover topic labels for FREX words.</p>
<div class="wrap"><table><tr><th>Topic</th>{header}</tr>{''.join(rows)}</table></div>
<h2>Topic Labels</h2><ol>{legend}</ol></body></html>"""
    html_path.write_text(output, encoding="utf-8")
    print(f"Rewrote STM K={k} labels and heatmap")


def main() -> None:
    for k in [10, 15, 20, 25, 30]:
        if (TABLES_DIR / f"stm_K{k}_topic_importance.csv").exists():
            rewrite_k(k)


if __name__ == "__main__":
    main()
