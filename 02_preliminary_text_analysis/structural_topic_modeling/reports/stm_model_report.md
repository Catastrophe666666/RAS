# STM Model Report

## Models Tested
- K values: 10, 15, 20, 25, 30

- K=10: mean semantic coherence=-35.735, mean exclusivity=9.545
- K=15: mean semantic coherence=-41.499, mean exclusivity=9.73
- K=20: mean semantic coherence=-43.293, mean exclusivity=9.755
- K=25: mean semantic coherence=-44.634, mean exclusivity=9.788
- K=30: mean semantic coherence=-46.16, mean exclusivity=9.789

## Inputs and Text Preparation
- Model document cache: C:\ras_text_analysis\files_for_text_analysis/outputs/topic_modeling/data/topic_modeling_model_documents.csv
- STM uses `model_text`, already prepared with shared stopword removal, tokenizer, and document-frequency filtering.
- Source path columns are retained in `stm_K*_document_topics.csv` for manual version checks.

## Candidate Model
- Diagnostic candidate: STM K=30

## Time and Importance Outputs
- `stm_K*_topic_by_decade_long.csv`: long-form topic prevalence by decade, including mean topic proportion and article coverage.
- `stm_K*_topic_by_decade_heatmap.html`: interactive-readable HTML heatmap sorted by topic importance.
- `stm_K*_topic_importance.csv`: topic ranking with article coverage, topic size, and interpretability components.
- `stm_K*_topic_overall_content_ranked.csv`: topic words and importance metrics ordered by importance.
- Article coverage threshold for STM topics: theta >= 0.05.
- STM interpretability is operationalized from normalized semantic coherence and exclusivity.

Use `stm_K*_topic_trends.png` and metadata tables to identify topics that increase or decrease over time and topics associated with L1/L2 categories.
Final labels should be assigned by the researcher.
