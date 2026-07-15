# BERTopic Report

## Pipeline
- Embedding text: cleaned readable article/paragraph/chunk text.
- Topic representation text: filtered text with English, RAS boilerplate, and OCR stopwords removed.
- Embedding model: `paraphrase-multilingual-MiniLM-L12-v2`.
- Stopwords used: 363; saved to `data/bertopic_custom_stopwords.txt`.
- Vectorizer: ngram_range=(1, 2), min_df=5, max_df=0.6, token_pattern=`(?u)\b[a-zA-Z][a-zA-Z'-]{2,}\b`.
- c-TF-IDF: bm25_weighting=True, reduce_frequent_words=True.
- UMAP: n_neighbors=20, n_components=5, min_dist=0.0.
- Barchart: top_n_topics=30, n_words=10, include_all=False.
- Outlier reduction attempted: True.

## Results
- chunk_dual_stopworded: level=chunk, docs=14,439, topics including -1=33, outliers=0 (0.0%), prefix=`bertopic_chunk_dual_stopworded`

## Interpretation Notes
- The legacy `bertopic_*.csv/html` files point to the primary improved variant.
- Use variant-prefixed outputs to compare chunk, paragraph, English-only, and default-like behavior.
- Historical OCR noise can still form topics; inspect representative chunks before assigning labels.
- Multilingual texts are retained unless an English-only variant is selected.
- Topic importance is reported with three retained components: article coverage, topic size, and interpretability.
- Interpretability is operationalized as top-5 c-TF-IDF weight concentration within the top-15 topic words.
- Topic-over-time outputs include detailed decade tables and an HTML heatmap sorted by topic importance.
