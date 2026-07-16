# Cross-Model Comparison Report

## Available Models
bertopic, bertopic_chunk_dual_stopworded, corex_anchored, corex_unanchored, lda, nmf, stm

## Alignment Method
Topics are compared by overlap among top words. Topic numbers are not treated as comparable across models.

- Topics available for review: 526
- Cross-model alignments with any shared words: 23,021
- Stronger alignments with Jaccard >= 0.20: 2,311

## Human Review
Use `topic_interpretation_template.csv` to assign researcher labels, merge/drop decisions, and interpretive notes.

## Cautions
- OCR artifacts can align across models and should be flagged rather than interpreted substantively.
- BERTopic-only and anchored-CorEx-only topics require close reading of representative documents.
