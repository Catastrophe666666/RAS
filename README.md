# RAS Journal Data

This repository contains cleaned datasets and interactive network visualizations for a digital humanities project on the journal materials of the Royal Asiatic Society (RAS), covering the period from **1858 to 1948**.

The project organizes article-level metadata into author, subject, and year-based tables, and further transforms the data into network-ready formats for analyzing the historical development of scholarly attention, subject categories, and author–subject relationships over time.

## Project Overview

The repository is designed to support exploratory analysis of English-language scholarship related to China and Asia from the late Qing period through the Republican era. The data can be used to examine questions such as:

* How did the thematic focus of RAS journal articles change between 1858 and 1948?
* Which subject areas became more or less prominent over time?
* How were authors connected to different subject categories?
* Which subjects occupied more central positions in the author–subject network?
* How can historical journal metadata be represented through network analysis and time-window visualization?

## Repository Structure

```text
RAS/
├── author_subject_year_tables/
│   ├── RAS_author_subject_year_1858-1948_utf8_2levsubjects.csv
│   └── RAS_author_subject_year_1858-1948_utf8_simplified_before_lcsh_namecorrected.csv
│
├── Network_Attributes/
│   ├── network_edges_article_level_lcsh_namecorrected.csv
│   ├── network_edges_author_subject_time_lcsh_namecorrected.csv
│   ├── network_nodes_author_subject_lcsh_namecorrected.csv
│   ├── subject_diversity_lcsh_namecorrected.csv
│   ├── subject_mapping_lcsh_review_namecorrected.csv
│   └── subject_network_centrality_time_lcsh_namecorrected.csv
│
├── Network_Html/
│   ├── author_subject_network_lcsh_interactive_namecorrected.html
│   └── subject_time_evolution_lcsh_namecorrected.html
│
└── README.md
```

### Folder Description

* `author_subject_year_tables/`
  Contains the main author–subject–year datasets for the RAS journal materials from 1858 to 1948. These files preserve article-level metadata and subject classification information.

* `Network_Attributes/`
  Contains network-ready datasets, including author–subject edge lists, node attributes, subject diversity measures, subject mapping files, and subject centrality measures over time.

* `Network_Html/`
  Contains interactive HTML visualizations for exploring the author–subject network and the evolution of subject categories across time.

* `README.md`
  Provides an overview of the project, dataset structure, possible uses, and suggested research directions.

## Possible Uses

This dataset can support:

* Digital humanities research
* Historical bibliography analysis
* Subject classification analysis
* Network analysis of authors and topics
* Time-series visualization of scholarly attention
* Exploratory research on the development of Anglophone scholarship on China and Asia

## Suggested Workflow

1. Start with the article-level tables in `author_subject_year_tables/`.
2. Use `Time_Window/` files to examine subject changes over time.
3. Use `Network_Attributes/` files for network analysis in R, Python, Gephi, or other visualization tools.
4. Open the files in `Network_Html/` for interactive exploration.

## Example Research Questions

This repository can be used to explore:

* Which subject categories dominated RAS journal publications in different periods?
* Did the journal’s scholarly attention shift from antiquarian, linguistic, or religious topics toward modern political, social, or economic concerns?
* Which authors were associated with the widest range of subject areas?
* Which subject categories served as bridges between different authors or scholarly communities?
* How did the structure of author–subject relationships change from the nineteenth century to the early twentieth century?

## Notes

The datasets have gone through subject classification, name correction, and LCSH-related review. However, because the source material comes from historical journal records, users should still treat the data as a research dataset that may require further checking for specific scholarly claims.


No license has been specified yet. Please contact the repository owner before reusing, redistributing, or publishing the dataset.
