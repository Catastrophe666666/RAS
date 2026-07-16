# STM Model Report

## Models Tested
- K values: 10, 15, 20, 25, 30

- K=10: mean semantic coherence=-40.922, mean exclusivity=9.556
- K=15: mean semantic coherence=-43.293, mean exclusivity=9.719
- K=20: mean semantic coherence=-44.445, mean exclusivity=9.758
- K=25: mean semantic coherence=-48.191, mean exclusivity=9.762
- K=30: mean semantic coherence=-47.951, mean exclusivity=9.779

## Inputs and Text Preparation
- Model document cache: C:\ras_text_analysis\files_for_text_analysis/outputs/topic_modeling/data/topic_modeling_model_documents.csv
- STM uses `model_text`, already prepared with shared stopword removal, tokenizer, and document-frequency filtering.
- English-only STM: TRUE
- Extra STM stopwords: C:\ras_text_analysis\files_for_text_analysis/outputs/topic_modeling/data/stm_extra_stopwords.txt
- Source path columns are retained in `stm_K*_document_topics.csv` for manual version checks.

## Candidate Model
- Balanced diagnostic candidate: STM K=20
- For the current diagnostics, K=15 is recommended as the main model: it is the elbow where exclusivity improves substantially over K=10, while gains after K=15 are small and semantic coherence continues to deteriorate.
- K=10 should be kept as a simpler robustness check if maximum semantic coherence is prioritized.

## Time and Importance Outputs
- `stm_K*_topic_by_decade_long.csv`: long-form topic prevalence by decade, including mean topic proportion and article coverage.
- `stm_K*_topic_by_decade_heatmap.html`: interactive-readable HTML heatmap sorted by topic importance.
- `stm_K*_topic_importance.csv`: topic ranking with article coverage, topic size, and interpretability components.
- `stm_K*_topic_overall_content_ranked.csv`: topic words and importance metrics ordered by importance.
- `stm_K*_representative_articles.csv`: high-theta representative articles for each topic.
- `stm_K*_topic_by_L1_long.csv`: topic prevalence by individual L1 subject.
- `stm_K*_topic_labels_improved.csv`: labels combining FREX, probability words, and representative titles.
- Article coverage threshold for STM topics: theta >= 0.05.
- STM interpretability is operationalized from normalized semantic coherence and exclusivity.

Use `stm_K*_topic_trends.png` and metadata tables to identify topics that increase or decrease over time and topics associated with L1/L2 categories.
Final labels should be assigned by the researcher.
