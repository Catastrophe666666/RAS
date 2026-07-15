# Workflow Overview

```text
ras_subject_expanded.csv + articles/*.txt + hit_stopwords.txt
        -> build_ras_text_dataset.py
article_text_master_dataset_v2_header_removed.csv
        -> build_topic_modeling_dataset.py
topic_modeling_articles.csv
topic_modeling_chunks.csv
topic_modeling_model_documents.csv
        -> TF-IDF / BERTopic / STM
```

The first stage links metadata to article text and produces the master text dataset. The second stage creates topic-modeling cache files. The cache files are not stored in this repository because they can be regenerated.

