# RAS Journal Digital Analysis

This repository preserves the current handoff package for the digital analysis of the Journal of the North China Branch of the Royal Asiatic Society.

The goal is continuity: a future researcher should be able to understand the current data state, inspect preliminary outputs, and rerun or extend the workflow without sorting through the full local working directory.

## Repository Structure

- `00_project_overview/`: project summary, current progress, workflow, future work, and notes on the China Journal.
- `01_processed_data/`: core metadata and the master article-text dataset used for later text analysis.
- `02_preliminary_text_analysis/`: preliminary social network, TF-IDF, BERTopic, and STM outputs.
- `03_scripts/`: scripts used for extraction, data processing, topic modeling, and network analysis.

## Minimal Data Strategy

This repository intentionally keeps a lightweight handoff version. It does not upload per-article cleaned txt files or large topic-modeling cache files. Instead, it keeps:

- `article_text_master_dataset_v2_header_removed.csv`
- metadata files needed to understand article identity and subject categories
- analysis scripts
- key output tables, reports, and visualizations

Large regenerated cache files such as `topic_modeling_articles.csv`, `topic_modeling_chunks.csv`, and `topic_modeling_model_documents.csv` are excluded from the repository.

## Main Workflow

```text
1858_1948_full_subject.csv
        -> corrected LCSH / subject-expanded mapping
ras_subject_expanded.csv
        -> social_network outputs
        -> build_ras_text_dataset.py + articles/*.txt + hit_stopwords.txt
article_text_master_dataset_v2_header_removed.csv
        -> build_topic_modeling_dataset.py
topic_modeling_articles.csv
topic_modeling_chunks.csv
topic_modeling_model_documents.csv
        -> TF-IDF / BERTopic / STM analysis
```

The large topic-modeling cache files are not stored here, but they can be regenerated from the master dataset and scripts.

