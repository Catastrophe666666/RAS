#!/usr/bin/env Rscript

# Structural Topic Modeling for the RAS article corpus.
# Run after scripts/build_topic_modeling_dataset.py.

script_path <- tryCatch(normalizePath(sys.frame(1)$ofile), error = function(e) NA_character_)
if (is.na(script_path)) {
  script_path <- normalizePath(commandArgs(trailingOnly = FALSE)[grep("--file=", commandArgs(trailingOnly = FALSE))[1]], mustWork = FALSE)
  script_path <- sub("^--file=", "", script_path)
}
script_dir <- ifelse(is.na(script_path) || script_path == "", getwd(), dirname(script_path))
project_dir <- normalizePath(file.path(script_dir, ".."), mustWork = FALSE)
outputs_dir <- file.path(project_dir, "outputs", "topic_modeling")
data_dir <- file.path(outputs_dir, "data")
tables_dir <- file.path(outputs_dir, "tables")
figures_dir <- file.path(outputs_dir, "figures")
reports_dir <- file.path(outputs_dir, "reports")

dir.create(tables_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(reports_dir, recursive = TRUE, showWarnings = FALSE)

required <- c("stm", "readr", "dplyr")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
report_path <- file.path(reports_dir, "stm_model_report.md")

if (length(missing) > 0) {
  writeLines(c(
    "# STM Model Report",
    "",
    "STM was not run because required R packages are unavailable:",
    paste(missing, collapse = ", "),
    "",
    "Install these packages in R, then rerun this script."
  ), report_path)
  quit(status = 0)
}

library(stm)
library(readr)
library(dplyr)

topic_threshold <- 0.05
english_only <- TRUE
extra_stopword_path <- file.path(data_dir, "stm_extra_stopwords.txt")

load_extra_stopwords <- function(path) {
  if (!file.exists(path)) {
    writeLines(c(
      "# Add one extra STM stopword per line.",
      "# Use this for OCR noise, romanization fragments, and terms that reduce interpretability.",
      "shih", "ssu", "chih", "chien", "huang", "wang", "hsien", "kuei"
    ), path)
  }
  values <- readLines(path, warn = FALSE, encoding = "UTF-8")
  values <- trimws(tolower(values))
  values <- values[values != "" & !startsWith(values, "#")]
  unique(values)
}

remove_extra_stopwords <- function(text, stopwords) {
  tokens <- unlist(strsplit(as.character(text), "\\s+"))
  tokens <- tokens[tokens != ""]
  tokens <- tokens[!(tolower(tokens) %in% stopwords)]
  paste(tokens, collapse = " ")
}

normalize_score <- function(x) {
  x <- as.numeric(x)
  if (length(x) == 0 || all(is.na(x))) {
    return(rep(0, length(x)))
  }
  lo <- min(x, na.rm = TRUE)
  hi <- max(x, na.rm = TRUE)
  if (hi <= lo) {
    return(ifelse(hi > 0, 1, 0))
  }
  (x - lo) / (hi - lo)
}

topic_label <- function(words, target_topic_id, n = 8, target_word_type = "frex") {
  words %>%
    filter(.data$topic_id == target_topic_id, .data$word_type == target_word_type) %>%
    arrange(.data$rank) %>%
    slice_head(n = n) %>%
    pull(.data$word) %>%
    paste(collapse = ", ")
}

html_escape <- function(x) {
  x <- as.character(x)
  x <- gsub("&", "&amp;", x, fixed = TRUE)
  x <- gsub("<", "&lt;", x, fixed = TRUE)
  x <- gsub(">", "&gt;", x, fixed = TRUE)
  x <- gsub("\"", "&quot;", x, fixed = TRUE)
  x <- gsub("'", "&#39;", x, fixed = TRUE)
  x
}

write_heatmap_html <- function(heatmap, summary, path, title) {
  if (nrow(heatmap) == 0) {
    writeLines(c("<html><body>", paste0("<h1>", title, "</h1>"), "<p>No topic-time data.</p>", "</body></html>"), path)
    return()
  }
  top_topics <- summary %>% arrange(.data$importance_rank) %>% slice_head(n = 30) %>% pull(.data$topic_id)
  heatmap <- heatmap %>% filter(.data$topic_id %in% top_topics)
  decades <- sort(unique(as.character(heatmap$decade)))
  max_value <- max(heatmap$mean_topic_proportion, na.rm = TRUE)
  label_lookup <- setNames(summary$topic_words, summary$topic_id)
  rows <- c()
  for (topic in top_topics) {
    topic_rows <- heatmap %>% filter(.data$topic_id == topic)
    cells <- c(paste0("<th title='", html_escape(label_lookup[as.character(topic)]), "'>Topic ", topic, "</th>"))
    for (decade_value in decades) {
      value <- topic_rows %>% filter(as.character(.data$decade) == decade_value) %>% pull(.data$mean_topic_proportion)
      if (length(value) == 0) value <- 0
      intensity <- ifelse(max_value <= 0, 0, min(value / max_value, 1))
      color <- paste0("rgb(", 255 - round(170 * intensity), ",", 255 - round(120 * intensity), ",", 255 - round(60 * intensity), ")")
      cells <- c(cells, paste0("<td style='background:", color, "'>", sprintf("%.3f", value), "</td>"))
    }
    rows <- c(rows, paste0("<tr>", paste(cells, collapse = ""), "</tr>"))
  }
  legend <- summary %>%
    arrange(.data$importance_rank) %>%
    slice_head(n = 30) %>%
    mutate(line = paste0("<li><b>Topic ", .data$topic_id, "</b>: ", html_escape(.data$topic_words), "</li>")) %>%
    pull(.data$line)
  html <- c(
    "<!doctype html><html><head><meta charset='utf-8'>",
    paste0("<title>", title, "</title>"),
    "<style>body{font-family:Arial,sans-serif;margin:24px} table{border-collapse:collapse} th,td{border:1px solid #ddd;padding:6px 8px;font-size:12px} th{background:#f5f5f5;position:sticky;top:0} td{text-align:right}.wrap{overflow:auto;max-height:75vh}</style>",
    "</head><body>",
    paste0("<h1>", title, "</h1>"),
    "<p>Cell values are mean STM topic proportions by decade. Hover topic labels for FREX words.</p>",
    "<div class='wrap'><table>",
    paste0("<tr><th>Topic</th>", paste0("<th>", decades, "</th>", collapse = ""), "</tr>"),
    rows,
    "</table></div><h2>Topic Labels</h2><ol>",
    legend,
    "</ol></body></html>"
  )
  writeLines(html, path)
}

articles <- read_csv(file.path(data_dir, "topic_modeling_model_documents.csv"), show_col_types = FALSE)
articles <- articles %>%
  filter(is.na(model_include) | model_include == TRUE | model_include == "TRUE") %>%
  filter(!is.na(model_text), nchar(model_text) > 0) %>%
  mutate(
    year = as.numeric(year),
    decade = as.factor(decade),
    L1 = as.factor(ifelse(is.na(L1) | L1 == "", "Unknown", L1)),
    L2 = as.factor(ifelse(is.na(L2) | L2 == "", "Unknown", L2)),
    language = as.factor(ifelse(is.na(language) | language == "", "unknown", language))
  )

if (english_only) {
  articles <- articles %>% filter(tolower(as.character(language)) %in% c("en", "english"))
}

extra_stopwords <- load_extra_stopwords(extra_stopword_path)
if (length(extra_stopwords) > 0) {
  articles$model_text <- vapply(articles$model_text, remove_extra_stopwords, character(1), stopwords = extra_stopwords)
  articles <- articles %>% filter(!is.na(model_text), nchar(model_text) > 0)
}

processed <- textProcessor(
  documents = articles$model_text,
  metadata = articles,
  lowercase = FALSE,
  removestopwords = FALSE,
  removenumbers = FALSE,
  removepunctuation = FALSE,
  stem = FALSE,
  wordLengths = c(3, Inf)
)

prep <- prepDocuments(processed$documents, processed$vocab, processed$meta, lower.thresh = 1)
docs <- prep$documents
vocab <- prep$vocab
meta <- prep$meta

k_values <- c(10, 15, 20, 25, 30)
diagnostics <- data.frame()
report_lines <- c("# STM Model Report", "", "## Models Tested", paste("- K values:", paste(k_values, collapse = ", ")), "")

for (k in k_values) {
  set.seed(42)
  prevalence_formula <- ~ s(year) + L1
  model <- stm(
    documents = docs,
    vocab = vocab,
    K = k,
    prevalence = prevalence_formula,
    data = meta,
    init.type = "Spectral",
    max.em.its = 75
  )

  label <- labelTopics(model, n = 20)
  words <- data.frame()
  for (topic_idx in seq_len(k)) {
    words <- rbind(words, data.frame(
      model = "stm",
      K = k,
      topic_id = topic_idx - 1,
      rank = seq_len(ncol(label$prob)),
      word = label$prob[topic_idx, ],
      word_type = "prob",
      stringsAsFactors = FALSE
    ))
    words <- rbind(words, data.frame(
      model = "stm",
      K = k,
      topic_id = topic_idx - 1,
      rank = seq_len(ncol(label$frex)),
      word = label$frex[topic_idx, ],
      word_type = "frex",
      stringsAsFactors = FALSE
    ))
  }
  write_csv(words, file.path(tables_dir, paste0("stm_K", k, "_topic_words.csv")))

  theta <- as.data.frame(model$theta)
  names(theta) <- paste0("topic_", seq_len(k) - 1)
  doc_topics <- bind_cols(
    meta %>% select(
      master_id, filename, title, author, year, decade, L1, L2, language,
      cleaned_full_text_path, analysis_text_path, model_source_column,
      model_token_count_before_df, model_token_count_after_df
    ),
    theta
  )
  write_csv(doc_topics, file.path(tables_dir, paste0("stm_K", k, "_document_topics.csv")))

  by_decade <- doc_topics %>% group_by(decade) %>% summarise(across(starts_with("topic_"), mean, na.rm = TRUE), .groups = "drop")
  write_csv(by_decade, file.path(tables_dir, paste0("stm_K", k, "_topic_by_decade.csv")))

  sem <- semanticCoherence(model, docs)
  excl <- exclusivity(model)
  topic_summary <- data.frame()
  topic_by_decade_long <- data.frame()
  theta_matrix <- as.matrix(theta)
  sem_norm <- normalize_score(sem)
  excl_norm <- normalize_score(excl)
  for (topic_idx in seq_len(k)) {
    topic_col <- paste0("topic_", topic_idx - 1)
    topic_values <- theta_matrix[, topic_idx]
    topic_summary <- rbind(topic_summary, data.frame(
      model = "stm",
      K = k,
      topic_id = topic_idx - 1,
      topic_size = sum(topic_values, na.rm = TRUE),
      topic_size_share = mean(topic_values, na.rm = TRUE),
      article_count = sum(topic_values >= topic_threshold, na.rm = TRUE),
      article_coverage = mean(topic_values >= topic_threshold, na.rm = TRUE),
      semantic_coherence = sem[topic_idx],
      exclusivity = excl[topic_idx],
      interpretability = mean(c(sem_norm[topic_idx], excl_norm[topic_idx]), na.rm = TRUE),
      topic_words = topic_label(words, topic_idx - 1, 12, "frex"),
      prob_words = topic_label(words, topic_idx - 1, 12, "prob"),
      stringsAsFactors = FALSE
    ))
    topic_by_decade_long <- rbind(
      topic_by_decade_long,
      doc_topics %>%
        group_by(decade) %>%
        summarise(
          model = "stm",
          K = k,
          topic_id = topic_idx - 1,
          mean_topic_proportion = mean(.data[[topic_col]], na.rm = TRUE),
          article_count = sum(.data[[topic_col]] >= topic_threshold, na.rm = TRUE),
          article_coverage = mean(.data[[topic_col]] >= topic_threshold, na.rm = TRUE),
          topic_words = topic_label(words, topic_idx - 1, 8, "frex"),
          .groups = "drop"
        )
    )
  }
  topic_summary <- topic_summary %>%
    mutate(
      coverage_score_norm = normalize_score(.data$article_coverage),
      size_score_norm = normalize_score(.data$topic_size),
      interpretability_score_norm = normalize_score(.data$interpretability),
      importance_score = (.data$coverage_score_norm + .data$size_score_norm + .data$interpretability_score_norm) / 3
    ) %>%
    arrange(desc(.data$importance_score), desc(.data$article_coverage), desc(.data$topic_size)) %>%
    mutate(importance_rank = row_number()) %>%
    select(
      importance_rank, everything()
    )
  write_csv(topic_summary, file.path(tables_dir, paste0("stm_K", k, "_topic_importance.csv")))
  write_csv(topic_summary %>% select(importance_rank, topic_id, importance_score, article_coverage, article_count, topic_size, topic_size_share, interpretability, semantic_coherence, exclusivity, topic_words, prob_words),
            file.path(tables_dir, paste0("stm_K", k, "_topic_overall_content_ranked.csv")))
  write_csv(topic_by_decade_long, file.path(tables_dir, paste0("stm_K", k, "_topic_by_decade_long.csv")))
  write_heatmap_html(
    topic_by_decade_long,
    topic_summary,
    file.path(figures_dir, paste0("stm_K", k, "_topic_by_decade_heatmap.html")),
    paste("STM K", k, "Topic Prevalence by Decade")
  )

  png(file.path(figures_dir, paste0("stm_K", k, "_diagnostics.png")), width = 1400, height = 900)
  plot(model, type = "summary", main = paste("STM K", k, "Topic Summary"))
  dev.off()

  png(file.path(figures_dir, paste0("stm_K", k, "_topic_trends.png")), width = 1400, height = 900)
  estimate <- estimateEffect(1:k ~ s(year), model, meta = meta)
  plot(estimate, "year", method = "continuous", topics = 1:min(k, 9), printlegend = TRUE)
  dev.off()

  diagnostics <- rbind(diagnostics, data.frame(
    K = k,
    semantic_coherence_mean = mean(sem),
    exclusivity_mean = mean(excl)
  ))
  report_lines <- c(report_lines, paste0("- K=", k, ": mean semantic coherence=", round(mean(sem), 3), ", mean exclusivity=", round(mean(excl), 3)))
}

write_csv(diagnostics, file.path(tables_dir, "stm_diagnostics.csv"))
diagnostics <- diagnostics %>%
  mutate(
    coherence_norm = normalize_score(.data$semantic_coherence_mean),
    exclusivity_norm = normalize_score(.data$exclusivity_mean),
    balanced_score = (.data$coherence_norm + .data$exclusivity_norm) / 2,
    coherence_weighted_score = 0.6 * .data$coherence_norm + 0.4 * .data$exclusivity_norm
  )
write_csv(diagnostics, file.path(tables_dir, "stm_k_selection_scores.csv"))
best <- diagnostics %>% arrange(desc(.data$balanced_score), desc(.data$semantic_coherence_mean)) %>% slice(1)
report_lines <- c(
  report_lines,
  "",
  "## Inputs and Text Preparation",
  paste0("- Model document cache: ", file.path(data_dir, "topic_modeling_model_documents.csv")),
  "- STM uses `model_text`, already prepared with shared stopword removal, tokenizer, and document-frequency filtering.",
  paste0("- English-only STM: ", english_only),
  paste0("- Extra STM stopwords: ", extra_stopword_path),
  "- Source path columns are retained in `stm_K*_document_topics.csv` for manual version checks.",
  "",
  "## Candidate Model",
  paste0("- Balanced diagnostic candidate: STM K=", best$K),
  "- For the current diagnostics, K=15 is recommended as the main model: it is the elbow where exclusivity improves substantially over K=10, while gains after K=15 are small and semantic coherence continues to deteriorate.",
  "- K=10 should be kept as a simpler robustness check if maximum semantic coherence is prioritized.",
  "",
  "## Time and Importance Outputs",
  "- `stm_K*_topic_by_decade_long.csv`: long-form topic prevalence by decade, including mean topic proportion and article coverage.",
  "- `stm_K*_topic_by_decade_heatmap.html`: interactive-readable HTML heatmap sorted by topic importance.",
  "- `stm_K*_topic_importance.csv`: topic ranking with article coverage, topic size, and interpretability components.",
  "- `stm_K*_topic_overall_content_ranked.csv`: topic words and importance metrics ordered by importance.",
  "- `stm_K*_representative_articles.csv`: high-theta representative articles for each topic.",
  "- `stm_K*_topic_by_L1_long.csv`: topic prevalence by individual L1 subject.",
  "- `stm_K*_topic_labels_improved.csv`: labels combining FREX, probability words, and representative titles.",
  paste0("- Article coverage threshold for STM topics: theta >= ", topic_threshold, "."),
  "- STM interpretability is operationalized from normalized semantic coherence and exclusivity.",
  "",
  "Use `stm_K*_topic_trends.png` and metadata tables to identify topics that increase or decrease over time and topics associated with L1/L2 categories.",
  "Final labels should be assigned by the researcher."
)
writeLines(report_lines, report_path)

python_candidates <- c(
  file.path(Sys.getenv("CONDA_PREFIX"), "python.exe"),
  "C:/Users/31745/anaconda3/envs/ras_env/python.exe",
  Sys.which("python"),
  Sys.which("python3")
)
python_exe <- python_candidates[file.exists(python_candidates) | nzchar(python_candidates)][1]
postprocess_script <- file.path(script_dir, "postprocess_stm_outputs.py")
if (!is.na(python_exe) && nzchar(python_exe) && file.exists(postprocess_script)) {
  system2(python_exe, c("-X", "utf8", postprocess_script))
}
