# Social Network Analysis

This folder contains corrected author-subject network outputs based on `1858_1948_full_subject.csv` and the subject-expanded mapping used in the current RAS analysis.

- `visual/`: interactive HTML visualizations and summary SVG/HTML visuals.
- `tables/`: node, edge, time-window, subject distribution, diversity, title-topic/place, and centrality tables.
- `tables/database_export/`: MySQL/MariaDB-oriented CSV and SQL export files.

Key files:

- `../../01_processed_data/metadata/ras_subject_expanded.csv`: corrected article-subject expanded data.
- `tables/network_edges_author_subject_time_subject_expanded_namecorrected.csv`: author-subject-time edge weights.
- `tables/network_nodes_author_subject_subject_expanded_namecorrected.csv`: author and subject nodes.
- `visual/author_subject_network_newest.html`: interactive author-subject network.
- `visual/subject_time_evolution_newest.html`: interactive subject evolution page.

Notes:

- Unknown subjects are excluded from the corrected network and visual summaries.
- The 2026-07-17 rebuild keeps `inscriptions` and `miscellaneous` as reviewed network categories under `Inscriptions and Miscellaneous`.
- The final time window is normalized as `1938-1948`.
- Short legacy table names are retained as compatibility copies of the corrected subject-expanded outputs.

