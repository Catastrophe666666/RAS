# Data Dictionary

## Core Metadata

- `1858_1948_full_subject.csv`: article-level metadata and subject records across the RAS Journal period.
- `merge_index_and_master.csv`: combined metadata produced from index records and master article records.
- `ras_subject_expanded.csv`: main subject-expanded metadata used by the current network and text-analysis workflows.

## Master Text Dataset

- `article_text_master_dataset_v2_header_removed.csv`: article-level text dataset after header removal and metadata matching. This is the retained lightweight replacement for uploading all cleaned txt files and topic-modeling cache files.

## Social Network Outputs

- `network_nodes_author_subject.csv`: node table for authors and subjects.
- `network_edges_article_level.csv`: article-level author-subject edges.
- `network_edges_author_subject_time.csv`: time-windowed author-subject edges.
- `subject_l1_time.csv`: first-level subject distribution over time.
- `subject_l2_time.csv`: second-level subject distribution over time.
- `subject_distribution_time_window.csv`: subject distribution by time window.
- `subject_diversity.csv`: subject diversity measures.
- `subject_network_centrality_time.csv`: centrality statistics by time window.

## Topic Modeling Outputs

- TF-IDF files summarize terms and phrases by overall corpus, decade, and subject category.
- BERTopic files summarize topic words, topic importance, topic prevalence by decade, and topic prevalence by first-level subject.
- STM files summarize K-selection, topic words, topic prevalence, document-topic assignments, and representative articles.

