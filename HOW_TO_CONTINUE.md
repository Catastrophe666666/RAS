# How to Continue

## 1. Start with the Data State

Read:

- `00_project_overview/workflow_overview.md`
- `01_processed_data/metadata/README.md`
- `01_processed_data/article_text_master_dataset/README.md`

The master text dataset is the recommended starting point for future text analysis.

## 2. Rebuild Topic Modeling Inputs if Needed

Use:

```text
03_scripts/data_processing/build_topic_modeling_dataset.py
```

This regenerates topic-modeling cache files from the master dataset. These cache files are intentionally not stored in the repository.

## 3. Extend Preliminary Analysis

Use:

- `03_scripts/topic_modeling/run_tfidf_exploration.py`
- `03_scripts/topic_modeling/run_bertopic_model.py`
- `03_scripts/topic_modeling/run_stm_topic_modeling.R`

For social network analysis, use:

- `03_scripts/network_analysis/author_subject_network_ras.Rmd`
- `03_scripts/network_analysis/build_subject_evolution.py`

## 4. China Journal Note

The China Journal has not yet been OCRed or structured. The current RAS workflow can serve as a methodological template for a future China Journal pipeline, beginning with OCR, article segmentation, metadata extraction, and validation.

