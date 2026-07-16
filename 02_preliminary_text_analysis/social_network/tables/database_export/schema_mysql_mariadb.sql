CREATE DATABASE IF NOT EXISTS ras_subject_analysis
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;
USE ras_subject_analysis;

DROP TABLE IF EXISTS article_title_terms;
DROP TABLE IF EXISTS article_title_places;
DROP TABLE IF EXISTS title_place_subject_time_stats;
DROP TABLE IF EXISTS title_place_time_stats;
DROP TABLE IF EXISTS title_place_name_mapping;
DROP TABLE IF EXISTS title_term_subject_time_stats;
DROP TABLE IF EXISTS title_tfidf_time_stats;
DROP TABLE IF EXISTS title_bigram_time_stats;
DROP TABLE IF EXISTS title_term_time_stats;
DROP TABLE IF EXISTS network_time_edges;
DROP TABLE IF EXISTS network_article_edges;
DROP TABLE IF EXISTS network_nodes;
DROP TABLE IF EXISTS subject_network_centrality_stats;
DROP TABLE IF EXISTS subject_diversity_stats;
DROP TABLE IF EXISTS subject_distribution_time_window;
DROP TABLE IF EXISTS subject_distribution_all_years;
DROP TABLE IF EXISTS subject_l2_time_stats;
DROP TABLE IF EXISTS subject_l1_time_stats;
DROP TABLE IF EXISTS article_subjects;
DROP TABLE IF EXISTS author_article_details;
DROP TABLE IF EXISTS author_article_stats_by_time_window;
DROP TABLE IF EXISTS author_article_stats;
DROP TABLE IF EXISTS article_authors;
DROP TABLE IF EXISTS time_windows;
DROP TABLE IF EXISTS subjects;
DROP TABLE IF EXISTS authors;
DROP TABLE IF EXISTS articles;

CREATE TABLE articles (
  master_id VARCHAR(32) PRIMARY KEY,
  year VARCHAR(32),
  volume VARCHAR(128),
  display_author VARCHAR(255),
  title TEXT,
  year_start INT,
  year_end INT,
  notes TEXT,
  time_window VARCHAR(16),
  INDEX idx_articles_year (year_start),
  INDEX idx_articles_window (time_window)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE authors (
  author_id VARCHAR(16) PRIMARY KEY,
  author_name VARCHAR(255) NOT NULL,
  UNIQUE KEY uq_author_name (author_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE article_authors (
  master_id VARCHAR(32) NOT NULL,
  author_id VARCHAR(16) NOT NULL,
  PRIMARY KEY (master_id, author_id),
  INDEX idx_article_authors_author (author_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE author_article_stats (
  author_name VARCHAR(255) NOT NULL,
  article_count INT,
  subject_count INT,
  first_year INT,
  last_year INT,
  INDEX idx_author_stats_name (author_name),
  INDEX idx_author_stats_count (article_count)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE author_article_stats_by_time_window (
  time_window VARCHAR(16),
  author_name VARCHAR(255) NOT NULL,
  article_count INT,
  subject_count INT,
  first_year INT,
  last_year INT,
  INDEX idx_author_window_stats_window (time_window),
  INDEX idx_author_window_stats_name (author_name),
  INDEX idx_author_window_stats_count (article_count)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE author_article_details (
  author_name VARCHAR(255) NOT NULL,
  master_id VARCHAR(32) NOT NULL,
  title TEXT,
  volume VARCHAR(128),
  year_start INT,
  year_end INT,
  time_window VARCHAR(16),
  subject_l1 VARCHAR(128),
  subject_l2 VARCHAR(128),
  INDEX idx_author_details_name (author_name),
  INDEX idx_author_details_window (time_window),
  INDEX idx_author_details_year (year_start),
  INDEX idx_author_details_master (master_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE subjects (
  subject_id VARCHAR(16) PRIMARY KEY,
  subject_l1 VARCHAR(128) NOT NULL,
  subject_l2 VARCHAR(128) NOT NULL,
  UNIQUE KEY uq_subject_pair (subject_l1, subject_l2)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE article_subjects (
  master_id VARCHAR(32) NOT NULL,
  subject_id VARCHAR(16) NOT NULL,
  subject_weight DOUBLE,
  subject_split VARCHAR(128),
  mapping_note TEXT,
  PRIMARY KEY (master_id, subject_id),
  INDEX idx_article_subjects_subject (subject_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE time_windows (
  time_window VARCHAR(16) PRIMARY KEY,
  start_year INT,
  end_year INT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE subject_l1_time_stats (
  time_window VARCHAR(16),
  subject_l1 VARCHAR(128),
  weighted_article_count DOUBLE,
  article_count INT,
  subject_l2_count INT,
  author_count INT,
  share DOUBLE,
  INDEX idx_l1_time (time_window),
  INDEX idx_l1_subject (subject_l1)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE subject_l2_time_stats (
  time_window VARCHAR(16),
  subject_l1 VARCHAR(128),
  subject_l2 VARCHAR(128),
  weighted_article_count DOUBLE,
  article_count INT,
  author_count INT,
  share DOUBLE,
  INDEX idx_l2_time (time_window),
  INDEX idx_l2_subject (subject_l2)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE subject_distribution_all_years (
  subject_l1 VARCHAR(128),
  subject_l2 VARCHAR(128),
  weighted_article_count DOUBLE,
  article_count INT,
  author_count INT,
  first_year INT,
  last_year INT,
  share DOUBLE,
  INDEX idx_all_subject_l1 (subject_l1),
  INDEX idx_all_subject_l2 (subject_l2)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE subject_distribution_time_window (
  time_window VARCHAR(16),
  subject_l1 VARCHAR(128),
  subject_l2 VARCHAR(128),
  weighted_article_count DOUBLE,
  article_count INT,
  author_count INT,
  first_year INT,
  last_year INT,
  share DOUBLE,
  INDEX idx_subject_dist_window (time_window),
  INDEX idx_subject_dist_window_l1 (subject_l1),
  INDEX idx_subject_dist_window_l2 (subject_l2)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE subject_diversity_stats (
  time_window VARCHAR(16),
  weighted_article_count DOUBLE,
  article_count INT,
  n_subject_l1 INT,
  n_subject_l2 INT,
  shannon_entropy DOUBLE,
  hhi DOUBLE,
  top3_share DOUBLE,
  top5_share DOUBLE,
  PRIMARY KEY (time_window)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE subject_network_centrality_stats (
  time_window VARCHAR(16),
  subject_l1 VARCHAR(128),
  subject_l2 VARCHAR(128),
  strength DOUBLE,
  article_count INT,
  author_count INT,
  degree INT,
  betweenness DOUBLE,
  INDEX idx_centrality_time (time_window),
  INDEX idx_centrality_subject (subject_l2)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE network_nodes (
  node_id VARCHAR(32),
  node_label VARCHAR(255),
  node_type VARCHAR(32),
  first_year INT,
  last_year INT,
  article_count INT,
  subject_count INT,
  subject_l1 VARCHAR(128),
  author_count INT,
  PRIMARY KEY (node_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE network_article_edges (
  from_node VARCHAR(32),
  to_node VARCHAR(32),
  author VARCHAR(255),
  subject_l2 VARCHAR(128),
  subject_l1 VARCHAR(128),
  master_id VARCHAR(32),
  title TEXT,
  year_start INT,
  year_end INT,
  time_window VARCHAR(16),
  subject_weight DOUBLE,
  subject_raw TEXT,
  INDEX idx_article_edges_window (time_window),
  INDEX idx_article_edges_subject (subject_l2)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE network_time_edges (
  from_node VARCHAR(32),
  to_node VARCHAR(32),
  author VARCHAR(255),
  subject_l2 VARCHAR(128),
  subject_l1 VARCHAR(128),
  time_window VARCHAR(16),
  article_count INT,
  weight DOUBLE,
  first_year INT,
  last_year INT,
  titles MEDIUMTEXT,
  INDEX idx_time_edges_window (time_window),
  INDEX idx_time_edges_subject (subject_l2)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE title_term_time_stats (
  time_window VARCHAR(16),
  term VARCHAR(128),
  term_count INT,
  article_count INT,
  share DOUBLE,
  INDEX idx_title_term_time (time_window),
  INDEX idx_title_term (term)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE title_bigram_time_stats (
  time_window VARCHAR(16),
  bigram VARCHAR(255),
  term_count INT,
  article_count INT,
  share DOUBLE,
  INDEX idx_title_bigram_time (time_window),
  INDEX idx_title_bigram (bigram)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE title_tfidf_time_stats (
  time_window VARCHAR(16),
  term VARCHAR(128),
  term_count INT,
  article_count INT,
  share DOUBLE,
  idf DOUBLE,
  tfidf DOUBLE,
  INDEX idx_title_tfidf_time (time_window),
  INDEX idx_title_tfidf_term (term)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE title_term_subject_time_stats (
  time_window VARCHAR(16),
  subject_l1 VARCHAR(128),
  subject_l2 VARCHAR(128),
  term VARCHAR(128),
  term_count INT,
  article_count INT,
  share DOUBLE,
  idf DOUBLE,
  tfidf DOUBLE,
  INDEX idx_title_subject_time (time_window),
  INDEX idx_title_subject_term (term),
  INDEX idx_title_subject_subject (subject_l2)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE article_title_terms (
  master_id VARCHAR(32),
  time_window VARCHAR(16),
  author_raw VARCHAR(255),
  title TEXT,
  year_start INT,
  term VARCHAR(128),
  INDEX idx_article_title_terms_master (master_id),
  INDEX idx_article_title_terms_term (term)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE title_place_time_stats (
  time_window VARCHAR(16),
  place_label VARCHAR(255),
  modern_name VARCHAR(128),
  historical_name VARCHAR(128),
  lat DOUBLE,
  lon DOUBLE,
  place_count INT,
  article_count INT,
  share DOUBLE,
  idf DOUBLE,
  tfidf DOUBLE,
  INDEX idx_place_time (time_window),
  INDEX idx_place_label (place_label)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE title_place_subject_time_stats (
  time_window VARCHAR(16),
  subject_l1 VARCHAR(128),
  subject_l2 VARCHAR(128),
  place_label VARCHAR(255),
  modern_name VARCHAR(128),
  historical_name VARCHAR(128),
  lat DOUBLE,
  lon DOUBLE,
  place_count INT,
  article_count INT,
  share DOUBLE,
  idf DOUBLE,
  tfidf DOUBLE,
  INDEX idx_place_subject_time (time_window),
  INDEX idx_place_subject_label (place_label),
  INDEX idx_place_subject_subject (subject_l2)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE article_title_places (
  master_id VARCHAR(32),
  time_window VARCHAR(16),
  author_raw VARCHAR(255),
  title TEXT,
  year_start INT,
  historical_name VARCHAR(128),
  modern_name VARCHAR(128),
  place_label VARCHAR(255),
  lat DOUBLE,
  lon DOUBLE,
  INDEX idx_article_places_master (master_id),
  INDEX idx_article_places_label (place_label)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE title_place_name_mapping (
  historical_name VARCHAR(128),
  modern_name VARCHAR(128),
  place_label VARCHAR(255),
  lat DOUBLE,
  lon DOUBLE,
  INDEX idx_place_mapping_label (place_label)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
