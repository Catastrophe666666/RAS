# TF-IDF Exploration and LDA Probe Report

## Corpus
- Documents analyzed: 590
- Documents excluded before article-level TF-IDF/LDA: 42
- Documents with post-filter model tokens: 590
- Overall TF-IDF terms saved: 100
- Tokenizer: Latin alphabet tokens, lowercased, length >= 3, plus bigrams/trigrams encoded with underscores.
- Stopword removal: RAS-first `hit_stopwords.txt`, RAS domain stopwords, and optional English stopwords when available.
- Document-frequency filter: min_df=5, max_df=0.6, max_features=12000.

## Source Path Samples
- `1858_01.txt`: readable=`C:\ras_text_analysis\files_for_text_analysis\outputs\cleaned_readable_texts_header_removed\1858_01.txt`; wordbag=`C:\ras_text_analysis\files_for_text_analysis\outputs\cleaned_wordbag_text_header_removed\1858_01.txt`
- `1858_02.txt`: readable=`C:\ras_text_analysis\files_for_text_analysis\outputs\cleaned_readable_texts_header_removed\1858_02.txt`; wordbag=`C:\ras_text_analysis\files_for_text_analysis\outputs\cleaned_wordbag_text_header_removed\1858_02.txt`
- `1858_03.txt`: readable=`C:\ras_text_analysis\files_for_text_analysis\outputs\cleaned_readable_texts_header_removed\1858_03.txt`; wordbag=`C:\ras_text_analysis\files_for_text_analysis\outputs\cleaned_wordbag_text_header_removed\1858_03.txt`
- `1858_04.txt`: readable=`C:\ras_text_analysis\files_for_text_analysis\outputs\cleaned_readable_texts_header_removed\1858_04.txt`; wordbag=`C:\ras_text_analysis\files_for_text_analysis\outputs\cleaned_wordbag_text_header_removed\1858_04.txt`
- `1858_05.txt`: readable=`C:\ras_text_analysis\files_for_text_analysis\outputs\cleaned_readable_texts_header_removed\1858_05.txt`; wordbag=`C:\ras_text_analysis\files_for_text_analysis\outputs\cleaned_wordbag_text_header_removed\1858_05.txt`
- `1858_06.txt`: readable=`C:\ras_text_analysis\files_for_text_analysis\outputs\cleaned_readable_texts_header_removed\1858_06.txt`; wordbag=`C:\ras_text_analysis\files_for_text_analysis\outputs\cleaned_wordbag_text_header_removed\1858_06.txt`
- `1858_07_record_of_occurence.txt`: readable=`C:\ras_text_analysis\files_for_text_analysis\outputs\cleaned_readable_texts_header_removed\1858_07_record_of_occurence.txt`; wordbag=`C:\ras_text_analysis\files_for_text_analysis\outputs\cleaned_wordbag_text_header_removed\1858_07_record_of_occurence.txt`
- `1859_01.txt`: readable=`C:\ras_text_analysis\files_for_text_analysis\outputs\cleaned_readable_texts_header_removed\1859_01.txt`; wordbag=`C:\ras_text_analysis\files_for_text_analysis\outputs\cleaned_wordbag_text_header_removed\1859_01.txt`

## LDA Probe
- LDA probe was not run.

## Chunk-Level LDA Probe
- Chunk-level LDA probe was not run.

TF-IDF is used as vocabulary diagnostics and stratified signal review. LDA uses count-style token lists from the shared model documents, not TF-IDF weights.
