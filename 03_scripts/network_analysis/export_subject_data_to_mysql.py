from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(r"C:\ras_text_analysis")
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "namecorrected_lcsh_outputs"
DB_DIR = OUT_DIR / "database_export"


FILES = {
    "expanded": OUT_DIR / "ras_subject_expanded.csv",
    "l1_time": OUT_DIR / "subject_l1_time_subject_expanded_namecorrected.csv",
    "l2_time": OUT_DIR / "subject_l2_time_subject_expanded_namecorrected.csv",
    "full_subject_distribution": OUT_DIR / "subject_distribution_all_years_subject_expanded_namecorrected.csv",
    "subject_distribution_time_window": OUT_DIR / "subject_distribution_time_window_subject_expanded_namecorrected.csv",
    "diversity": OUT_DIR / "subject_diversity_subject_expanded_namecorrected.csv",
    "centrality": OUT_DIR / "subject_network_centrality_time_subject_expanded_namecorrected.csv",
    "nodes": OUT_DIR / "network_nodes_author_subject_subject_expanded_namecorrected.csv",
    "edges_article": OUT_DIR / "network_edges_article_level_subject_expanded_namecorrected.csv",
    "edges_time": OUT_DIR / "network_edges_author_subject_time_subject_expanded_namecorrected.csv",
    "title_terms": OUT_DIR / "title_term_time_subject_expanded_namecorrected.csv",
    "title_bigrams": OUT_DIR / "title_bigram_time_subject_expanded_namecorrected.csv",
    "title_tfidf": OUT_DIR / "title_tfidf_time_subject_expanded_namecorrected.csv",
    "title_subject_terms": OUT_DIR / "title_term_subject_time_subject_expanded_namecorrected.csv",
    "article_title_terms": OUT_DIR / "article_title_terms_subject_expanded_namecorrected.csv",
    "title_places": OUT_DIR / "title_place_time_subject_expanded_namecorrected.csv",
    "title_subject_places": OUT_DIR / "title_place_subject_time_subject_expanded_namecorrected.csv",
    "article_title_places": OUT_DIR / "article_title_places_subject_expanded_namecorrected.csv",
    "title_place_mapping": OUT_DIR / "title_place_name_mapping_subject_expanded_namecorrected.csv",
}


def read_csv(name: str) -> pd.DataFrame:
    path = FILES[name]
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8")


def save_table(df: pd.DataFrame, table_name: str) -> Path:
    path = DB_DIR / f"{table_name}.csv"
    df = df.copy()
    df = df.where(pd.notna(df), "")
    df.to_csv(path, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)
    return path


def mysql_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "\\'")


def build_tables() -> dict[str, pd.DataFrame]:
    expanded = read_csv("expanded")
    expanded["year_start"] = pd.to_numeric(expanded["year_start"], errors="coerce")
    expanded["year_end"] = pd.to_numeric(expanded["year_end"], errors="coerce")
    expanded["subject_weight"] = pd.to_numeric(expanded["subject_weight"], errors="coerce").fillna(0)

    articles = (
        expanded[
            ["master_id", "year", "volume", "author_raw", "title", "year_start", "year_end", "notes", "time_window"]
        ]
        .drop_duplicates("master_id")
        .rename(columns={"author_raw": "display_author"})
        .sort_values(["year_start", "master_id"])
    )

    authors = (
        expanded[["author_raw"]]
        .dropna()
        .drop_duplicates()
        .rename(columns={"author_raw": "author_name"})
        .sort_values("author_name")
        .reset_index(drop=True)
    )
    authors.insert(0, "author_id", ["AUTH_%05d" % (i + 1) for i in range(len(authors))])

    article_authors = (
        expanded[["master_id", "author_raw"]]
        .dropna()
        .drop_duplicates()
        .merge(authors, left_on="author_raw", right_on="author_name", how="left")
        [["master_id", "author_id"]]
        .sort_values(["master_id", "author_id"])
    )

    author_article_details = (
        expanded[
            ["author_raw", "master_id", "title", "volume", "year_start", "year_end", "time_window", "subject_l1", "subject_l2"]
        ]
        .dropna(subset=["author_raw"])
        .drop_duplicates(["author_raw", "master_id", "subject_l2"])
        .rename(columns={"author_raw": "author_name"})
        .sort_values(["author_name", "year_start", "title", "subject_l2"])
    )

    author_article_stats = (
        expanded.dropna(subset=["author_raw"])
        .groupby("author_raw", as_index=False)
        .agg(
            article_count=("master_id", "nunique"),
            subject_count=("subject_l2", "nunique"),
            first_year=("year_start", "min"),
            last_year=("year_end", "max"),
        )
        .rename(columns={"author_raw": "author_name"})
        .sort_values(["article_count", "author_name"], ascending=[False, True])
    )

    author_article_stats_by_time_window = (
        expanded.dropna(subset=["author_raw"])
        .groupby(["time_window", "author_raw"], as_index=False)
        .agg(
            article_count=("master_id", "nunique"),
            subject_count=("subject_l2", "nunique"),
            first_year=("year_start", "min"),
            last_year=("year_end", "max"),
        )
        .rename(columns={"author_raw": "author_name"})
        .sort_values(["time_window", "article_count", "author_name"], ascending=[True, False, True])
    )

    subjects = (
        expanded[["subject_l1", "subject_l2"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["subject_l1", "subject_l2"])
        .reset_index(drop=True)
    )
    subjects.insert(0, "subject_id", ["SUBJ_%05d" % (i + 1) for i in range(len(subjects))])

    article_subjects = (
        expanded[
            [
                "master_id",
                "subject_l1",
                "subject_l2",
                "subject_weight",
                "subject_split",
                "mapping_note",
            ]
        ]
        .drop_duplicates(["master_id", "subject_l2"])
        .merge(subjects, on=["subject_l1", "subject_l2"], how="left")
        [["master_id", "subject_id", "subject_weight", "subject_split", "mapping_note"]]
        .sort_values(["master_id", "subject_id"])
    )

    time_windows = (
        expanded[["time_window"]]
        .drop_duplicates()
        .sort_values("time_window")
        .reset_index(drop=True)
    )
    time_windows["start_year"] = time_windows["time_window"].str.extract(r"^(\d{4})").astype(int)
    time_windows["end_year"] = time_windows["time_window"].str.extract(r"-(\d{4})$").astype(int)

    tables = {
        "articles": articles,
        "authors": authors,
        "article_authors": article_authors,
        "author_article_stats": author_article_stats,
        "author_article_stats_by_time_window": author_article_stats_by_time_window,
        "author_article_details": author_article_details,
        "subjects": subjects,
        "article_subjects": article_subjects,
        "time_windows": time_windows,
        "subject_l1_time_stats": read_csv("l1_time"),
        "subject_l2_time_stats": read_csv("l2_time"),
        "subject_distribution_all_years": read_csv("full_subject_distribution"),
        "subject_distribution_time_window": read_csv("subject_distribution_time_window"),
        "subject_diversity_stats": read_csv("diversity"),
        "subject_network_centrality_stats": read_csv("centrality"),
        "network_nodes": read_csv("nodes"),
        "network_article_edges": read_csv("edges_article"),
        "network_time_edges": read_csv("edges_time"),
        "title_term_time_stats": read_csv("title_terms"),
        "title_bigram_time_stats": read_csv("title_bigrams"),
        "title_tfidf_time_stats": read_csv("title_tfidf"),
        "title_term_subject_time_stats": read_csv("title_subject_terms"),
        "article_title_terms": read_csv("article_title_terms"),
        "title_place_time_stats": read_csv("title_places"),
        "title_place_subject_time_stats": read_csv("title_subject_places"),
        "article_title_places": read_csv("article_title_places"),
        "title_place_name_mapping": read_csv("title_place_mapping"),
    }
    return tables


SCHEMA_SQL = r"""
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
"""


def write_schema_and_loader(table_paths: dict[str, Path]) -> None:
    (DB_DIR / "schema_mysql_mariadb.sql").write_text(SCHEMA_SQL.strip() + "\n", encoding="utf-8")
    lines = [
        "USE ras_subject_analysis;",
        "SET NAMES utf8mb4;",
        "SET FOREIGN_KEY_CHECKS = 0;",
    ]
    for table, path in table_paths.items():
        lines.append(f"TRUNCATE TABLE {table};")
        lines.append(
            "LOAD DATA LOCAL INFILE '%s' INTO TABLE %s "
            "CHARACTER SET utf8mb4 FIELDS TERMINATED BY ',' ENCLOSED BY '\"' "
            "ESCAPED BY '\"' LINES TERMINATED BY '\\n' IGNORE 1 LINES;"
            % (mysql_path(path), table)
        )
    lines.append("SET FOREIGN_KEY_CHECKS = 1;")
    (DB_DIR / "load_data_mysql_mariadb.sql").write_text("\n".join(lines) + "\n", encoding="utf-8")


def try_import_with_pymysql() -> None:
    host = os.getenv("RAS_DB_HOST")
    user = os.getenv("RAS_DB_USER")
    password = os.getenv("RAS_DB_PASSWORD")
    database = os.getenv("RAS_DB_NAME", "ras_subject_analysis")
    if not host or not user:
        return
    try:
        import pymysql  # type: ignore
    except Exception:
        print("pymysql is not installed; wrote SQL and CSV export only.")
        return
    schema = (DB_DIR / "schema_mysql_mariadb.sql").read_text(encoding="utf-8")
    conn = pymysql.connect(host=host, user=user, password=password, local_infile=True, charset="utf8mb4", autocommit=True)
    with conn.cursor() as cur:
        for statement in schema.split(";"):
            if statement.strip():
                cur.execute(statement)
        cur.execute(f"USE {database}")
        loader = (DB_DIR / "load_data_mysql_mariadb.sql").read_text(encoding="utf-8")
        for statement in loader.split(";"):
            if statement.strip():
                cur.execute(statement)
    conn.close()
    print(f"Imported tables into MySQL/MariaDB database {database}.")


def main() -> None:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    tables = build_tables()
    table_paths = {name: save_table(df, name) for name, df in tables.items()}
    write_schema_and_loader(table_paths)
    try_import_with_pymysql()
    print(f"wrote database export to {DB_DIR}")
    print(f"tables={len(table_paths)}")


if __name__ == "__main__":
    main()
