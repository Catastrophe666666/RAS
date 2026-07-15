# Project Progress

## Completed

- Metadata has been reconciled from index-derived records and physical-copy article information.
- Article-level subject metadata has been projected into `ras_subject_expanded.csv`.
- A master article-text dataset has been generated: `article_text_master_dataset_v2_header_removed.csv`.
- Preliminary author-subject network outputs have been generated.
- Preliminary TF-IDF, BERTopic, and Structural Topic Modeling outputs have been generated.

## Current Status

The repository is a lightweight handoff package. It keeps the master text dataset and main output files, but excludes large intermediate caches and local-only working files.

## Not Included

- Raw PDF files
- OCR cache folders
- `.venv`, `node_modules`, and Python/R temporary files
- API key files
- Large vocabulary files
- Topic-modeling cache files that can be regenerated
- CorEx outputs, because CorEx did not become a completed analysis stage

