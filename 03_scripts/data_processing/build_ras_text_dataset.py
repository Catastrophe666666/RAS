#!/usr/bin/env python
"""
Build a conservative, multilingual-safe article text dataset for the RAS corpus.

Raw OCR files are never changed. All exclusions are represented as flags in CSV
outputs so the workflow is reproducible and reversible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DEFAULT_PROJECT_DIR = Path(r"C:\ras_text_analysis\files_for_text_analysis")
RUN_LABEL = "v2_header_removed"
READABLE_OUTPUT_DIR_NAME = "cleaned_readable_texts_header_removed"
WORDBAG_OUTPUT_DIR_NAME = "cleaned_wordbag_text_header_removed"
KEEP_OPENING_TITLE_AUTHOR = True
REMOVE_OPENING_METADATA_FROM_BODY = False

PAGE_MARKER_RE = re.compile(r"^\s*##\s*p\.?\s*\d+\s*\(#\d+\)\s*#+\s*$", re.I | re.M)
PAGE_MARKER_PAGE_RE = re.compile(r"p\.?\s*(\d+)", re.I)
ARTICLE_MARKER_RE = re.compile(r"^\s*ARTICLE\s+(?:[IVXLCDM]+|\d+)\.?\s*$", re.I)
STANDALONE_PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,4}\s*$")
TRAILING_PAGE_NUMBER_RE = re.compile(r"^(?P<header>.+?)\s+(?P<page>\d{1,4})\s*$")
LEADING_PAGE_NUMBER_RE = re.compile(r"^(?P<page>\d{1,4})\s+(?P<header>.+?)\s*$")
JOURNAL_FRONT_MATTER_TERMS = {
    "journal", "of the", "north china branch", "north-china branch",
    "royal asiatic society", "shanghai literary and scientific society",
}

EN_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "did", "do",
    "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here",
    "hers", "herself", "him", "himself", "his", "how", "i", "if", "in",
    "into", "is", "it", "its", "itself", "just", "me", "more", "most",
    "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over",
    "own", "s", "same", "she", "should", "so", "some", "such", "t",
    "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to",
    "too", "under", "until", "up", "very", "was", "we", "were", "what",
    "when", "where", "which", "while", "who", "whom", "why", "will",
    "with", "you", "your", "yours", "yourself", "yourselves",
}

FR_STOPWORDS = {
    "alors", "au", "aucuns", "aussi", "autre", "avant", "avec", "avoir",
    "bon", "car", "ce", "cela", "ces", "ceux", "chaque", "ci", "comme",
    "comment", "dans", "de", "des", "du", "dedans", "dehors", "depuis",
    "devrait", "doit", "donc", "dos", "droite", "elle", "elles", "en",
    "encore", "essai", "est", "et", "eu", "fait", "faites", "fois",
    "font", "hors", "ici", "il", "ils", "je", "juste", "la", "le",
    "les", "leur", "là", "ma", "maintenant", "mais", "mes", "mine",
    "moins", "mon", "mot", "même", "ni", "nommés", "notre", "nous",
    "nouveaux", "ou", "où", "par", "parce", "parole", "pas", "personnes",
    "peut", "peu", "pièce", "plupart", "pour", "pourquoi", "quand",
    "que", "quel", "quelle", "quelles", "quels", "qui", "sa", "sans",
    "ses", "seulement", "si", "sien", "son", "sont", "sous", "soyez",
    "sujet", "sur", "ta", "tandis", "tellement", "tels", "tes", "ton",
    "tous", "tout", "trop", "très", "tu", "voient", "vont", "votre",
    "vous", "vu", "ça", "étaient", "état", "étions", "été", "être",
}

DE_STOPWORDS = {
    "aber", "als", "am", "an", "auch", "auf", "aus", "bei", "bin",
    "bis", "bist", "da", "dadurch", "daher", "darum", "das", "daß",
    "dass", "dein", "deine", "dem", "den", "der", "des", "dessen",
    "deshalb", "die", "dies", "dieser", "dieses", "doch", "dort", "du",
    "durch", "ein", "eine", "einem", "einen", "einer", "eines", "er",
    "es", "euer", "eure", "für", "hatte", "hatten", "hattest", "hattet",
    "hier", "hinter", "ich", "ihr", "ihre", "im", "in", "ist", "ja",
    "jede", "jedem", "jeden", "jeder", "jedes", "jener", "jenes", "jetzt",
    "kann", "kannst", "können", "könnt", "machen", "mein", "meine",
    "mit", "muß", "musst", "müssen", "müßt", "nach", "nachdem", "nein",
    "nicht", "nun", "oder", "seid", "sein", "seine", "sich", "sie",
    "sind", "soll", "sollen", "sollst", "sollt", "sonst", "soweit",
    "sowie", "und", "unser", "unsere", "unter", "vom", "von", "vor",
    "wann", "warum", "was", "weiter", "weitere", "wenn", "wer", "werde",
    "werden", "werdet", "weshalb", "wie", "wieder", "wieso", "wir",
    "wird", "wirst", "wo", "woher", "wohin", "zu", "zum", "zur", "über",
}

TEXT_TYPE_RULES = [
    (("reviews_of_recent_books", "review_of_recent_books", "book_review", "reviews", "review"), "book_review_or_review_section"),
    (("index",), "index"),
    (("proceedings",), "proceedings"),
    (("minutes",), "minutes"),
    (("bibliography",), "bibliography"),
    (("contents", "table_of_contents", "front_matter", "preface", "title_page"), "front_matter"),
    (("appendix",), "appendix"),
    (("advertisement", "advertisements", "ads"), "advertisement"),
]

UNCERTAIN_SUFFIXES = ("misc", "notes_and_queries", "notes and queries", "literary_notes", "literary notes")


@dataclass
class TextRecord:
    text_id: str
    text_file: str
    raw_text_path: Path
    raw_text: str
    encoding: str
    file_size: int
    raw_char_count: int
    raw_line_count: int
    raw_text_checksum: str
    raw_norm_checksum: str
    filename_year_hint: str
    year_hint_start: int | None
    year_hint_end: int | None
    filename_suffix: str
    suggested_text_type: str
    final_text_type: str
    text_type_source: str
    exclude_from_article_analysis: bool
    exclusion_reason: str
    non_article_notes: str
    head_for_match: str
    head_norm: str
    head_tokens: set[str]
    head_lines_norm: list[str]
    detected_language: str
    language_confidence: float
    language_detection_method: str
    possible_mixed_language_flag: bool
    non_english_flag: bool
    chinese_flag: bool


@dataclass
class HeaderCleaningStats:
    generated_page_markers_removed: int = 0
    running_header_lines_removed: int = 0
    standalone_page_numbers_removed: int = 0
    journal_front_matter_removed: bool = False
    removed_lines: list[str] | None = None
    warnings: set[str] | None = None

    def __post_init__(self) -> None:
        if self.removed_lines is None:
            self.removed_lines = []
        if self.warnings is None:
            self.warnings = set()

    def remember(self, line: str) -> None:
        if len(self.removed_lines or []) < 10:
            self.removed_lines.append(line.strip())


def read_text_lossy(path: Path) -> tuple[str, str]:
    data = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace"), "utf-8-replace"


def sha1_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="replace")).hexdigest()


def normalize_unicode(value: str) -> str:
    value = str(value).replace("\r\n", "\n").replace("\r", "\n")
    value = value.replace("\ufeff", "").replace("\x00", "")
    value = unicodedata.normalize("NFKC", value)
    replacements = {
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-", "\u00ad": "",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


def normalize_for_match(value: str) -> str:
    value = unicodedata.normalize("NFKD", normalize_unicode(value))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower().replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def normalize_for_duplicate(value: str) -> str:
    value = normalize_unicode(value).lower()
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def extract_year_hint(filename: str) -> tuple[str, int | None, int | None]:
    years = [int(y) for y in re.findall(r"(?:18|19)\d{2}", filename)]
    if not years:
        return "", None, None
    start, end = min(years), max(years)
    return str(start) if start == end else f"{start}-{end}", start, end


def filename_suffix(filename: str) -> str:
    stem = Path(filename).stem
    stripped = re.sub(r"^(?:[A-Z]+_)?(?:18|19)\d{2}(?:-(?:18|19)\d{2})?_\d+_?", "", stem, flags=re.I)
    stripped = re.sub(r"^(?:18|19)\d{2}(?:-(?:18|19)\d{2})?_\d+_?", "", stripped, flags=re.I)
    return stripped.strip("_ -")


def infer_text_type(filename: str) -> tuple[str, str, str, bool, str, str]:
    lowered = Path(filename).stem.lower().replace("-", "_")
    lowered_spaced = lowered.replace("_", " ")
    for needles, text_type in TEXT_TYPE_RULES:
        if any(needle in lowered or needle in lowered_spaced for needle in needles):
            return (
                text_type,
                text_type,
                "filename_suffix_rule",
                True,
                "non_article_filename_suffix",
                "clear non-article suffix",
            )
    if any(suffix in lowered or suffix in lowered_spaced for suffix in UNCERTAIN_SUFFIXES):
        return (
            "uncertain",
            "uncertain",
            "filename_suffix_rule",
            True,
            "uncertain_text_type_conservative_exclusion",
            "filename suffix suggests a non-standard article-like section",
        )
    return (
        "article_candidate",
        "article_candidate",
        "filename_suffix_rule",
        False,
        "",
        "",
    )


def language_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿÄÖÜäöüß]+", text.lower())


def detect_language(text: str) -> tuple[str, float, str, bool, bool, bool]:
    sample = normalize_unicode(text[:25000])
    cjk_count = len(re.findall(r"[\u3400-\u9fff]", sample))
    alpha_tokens = language_tokens(sample)
    alpha_count = sum(len(tok) for tok in alpha_tokens)
    total_signal = cjk_count + alpha_count
    if total_signal == 0:
        return "unknown", 0.0, "heuristic_script_and_stopword", False, True, False

    cjk_ratio = cjk_count / total_signal
    if cjk_ratio >= 0.30:
        mixed = alpha_count > 250 and cjk_ratio < 0.75
        confidence = min(0.99, 0.60 + cjk_ratio)
        return "mixed" if mixed else "zh-Hant", round(confidence, 3), "heuristic_cjk_script", mixed, True, True

    token_set = set(alpha_tokens)
    scores = {
        "en": sum(1 for tok in alpha_tokens if tok in EN_STOPWORDS),
        "fr": sum(1 for tok in alpha_tokens if tok in FR_STOPWORDS),
        "de": sum(1 for tok in alpha_tokens if tok in DE_STOPWORDS),
    }
    accent_fr = len(re.findall(r"[àâçéèêëîïôûùüÿœ]", sample.lower()))
    accent_de = len(re.findall(r"[äöüß]", sample.lower()))
    scores["fr"] += min(80, accent_fr * 2)
    scores["de"] += min(80, accent_de * 3)
    common_english_terms = {"the", "of", "and", "china", "chinese", "royal", "asiatic", "society"}
    scores["en"] += sum(2 for tok in common_english_terms if tok in token_set)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_lang, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0
    if best_score < 8:
        return "unknown", 0.25, "heuristic_script_and_stopword", False, True, False
    confidence = (best_score + 1) / (sum(scores.values()) + 3)
    mixed = second_score > 0 and second_score / max(best_score, 1) >= 0.72 and best_score >= 15
    lang = "mixed" if mixed else best_lang
    return lang, round(min(0.99, max(0.35, confidence)), 3), "heuristic_script_and_stopword", mixed, lang != "en", False


def line_is_mostly_uppercase(line: str) -> bool:
    letters = re.findall(r"[A-Za-z]", line)
    if len(letters) < 6:
        return False
    uppercase = sum(1 for ch in letters if ch.isupper())
    return uppercase / len(letters) >= 0.72


def strip_trailing_page_number(line: str) -> tuple[str, bool]:
    match = TRAILING_PAGE_NUMBER_RE.match(line.strip())
    if not match:
        return line.strip(), False
    header = match.group("header").strip()
    if len(header) < 8:
        return line.strip(), False
    return header, True


def strip_leading_page_number(line: str) -> tuple[str, bool]:
    match = LEADING_PAGE_NUMBER_RE.match(line.strip())
    if not match:
        return line.strip(), False
    header = match.group("header").strip()
    if len(header) < 8:
        return line.strip(), False
    return header, True


def strip_edge_page_numbers(line: str) -> tuple[str, bool, bool]:
    without_leading, had_leading = strip_leading_page_number(line)
    without_trailing, had_trailing = strip_trailing_page_number(without_leading)
    return without_trailing.strip(), had_leading, had_trailing


def title_like_header(line: str, article_title: str | None) -> bool:
    if not article_title:
        return False
    line_norm = normalize_for_match(strip_edge_page_numbers(line)[0])
    title_norm = normalize_for_match(article_title)
    if not line_norm or not title_norm:
        return False
    if line_norm in title_norm or title_norm in line_norm:
        return True
    line_tokens = [tok for tok in line_norm.split() if len(tok) > 2]
    title_tokens = [tok for tok in title_norm.split() if len(tok) > 2]
    if not line_tokens or not title_tokens:
        return False
    overlap = sum(1 for tok in line_tokens if tok in set(title_tokens))
    return overlap / len(line_tokens) >= 0.70 and overlap >= min(3, len(line_tokens))


def likely_running_header(line: str, article_title: str | None, after_page_marker: bool) -> tuple[bool, str]:
    if not after_page_marker:
        return False, ""
    stripped = line.strip()
    if ARTICLE_MARKER_RE.match(stripped):
        return False, ""
    if not stripped or STANDALONE_PAGE_NUMBER_RE.match(stripped):
        return False, ""
    header_without_page, starts_with_page, ends_with_page = strip_edge_page_numbers(stripped)
    mostly_upper = line_is_mostly_uppercase(header_without_page)
    title_like = title_like_header(header_without_page, article_title)
    shortish = len(header_without_page) <= 140
    has_edge_page = starts_with_page or ends_with_page
    if shortish and (mostly_upper or title_like) and has_edge_page:
        return True, "uppercase_or_title_like_with_edge_page_number"
    if shortish and mostly_upper and title_like:
        return True, "uppercase_title_like"
    if shortish and mostly_upper and len(header_without_page.split()) >= 2:
        return True, "mostly_uppercase_after_page_marker"
    if shortish and title_like and len(header_without_page.split()) >= 3:
        return True, "title_like_after_page_marker"
    return False, ""


def remove_initial_journal_front_matter(lines: list[str], stats: HeaderCleaningStats) -> list[str]:
    search_limit = min(len(lines), 40)
    article_idx = None
    for idx in range(search_limit):
        if ARTICLE_MARKER_RE.match(lines[idx].strip()):
            article_idx = idx
            break
    if article_idx is None:
        head_norm = " ".join(normalize_for_match(line) for line in lines[:12])
        if "journal" in head_norm and "royal asiatic society" in head_norm:
            stats.warnings.add("front_matter_detected_but_article_marker_not_found")
        return lines
    prefix = lines[:article_idx]
    non_empty = [line.strip() for line in prefix if line.strip()]
    if not non_empty:
        return lines
    normalized = [normalize_for_match(line) for line in non_empty]
    boilerplate_hits = 0
    for norm in normalized:
        if any(term in norm for term in JOURNAL_FRONT_MATTER_TERMS):
            boilerplate_hits += 1
    mostly_boilerplate = boilerplate_hits >= 2 and boilerplate_hits / max(len(normalized), 1) >= 0.45
    if mostly_boilerplate:
        stats.journal_front_matter_removed = True
        stats.warnings.add("front_matter_removed")
        for line in non_empty:
            stats.remember(line)
        return lines[article_idx:]
    return lines


def page_number_from_marker(line: str) -> str:
    match = PAGE_MARKER_PAGE_RE.search(line)
    return match.group(1) if match else ""


def next_non_empty_indices(lines: list[str], start_idx: int, limit: int = 4) -> list[int]:
    indices = []
    idx = start_idx
    while idx < len(lines) and len(indices) < limit:
        if lines[idx].strip():
            indices.append(idx)
        idx += 1
    return indices


def consume_header_after_page_marker(
    lines: list[str],
    start_idx: int,
    marker_line: str,
    article_title: str | None,
    stats: HeaderCleaningStats,
) -> int:
    marker_page = page_number_from_marker(marker_line)
    candidates = next_non_empty_indices(lines, start_idx, limit=4)
    if not candidates:
        stats.warnings.add("unusual_header_pattern")
        return start_idx

    first_idx = candidates[0]
    first = lines[first_idx].strip()

    if any(ARTICLE_MARKER_RE.match(lines[candidate_idx].strip()) for candidate_idx in candidates):
        return start_idx

    if STANDALONE_PAGE_NUMBER_RE.match(first):
        if len(candidates) < 2:
            stats.warnings.add("unusual_header_pattern")
            return start_idx
        second_idx = candidates[1]
        second = lines[second_idx].strip()
        is_header, reason = likely_running_header(second, article_title, after_page_marker=True)
        exact_marker_page = bool(marker_page and first == marker_page)
        second_without_page = strip_edge_page_numbers(second)[0]
        uppercase_short = line_is_mostly_uppercase(second_without_page) and len(second_without_page) <= 140
        if is_header or (exact_marker_page and uppercase_short):
            stats.standalone_page_numbers_removed += 1
            stats.remember(lines[first_idx])
            stats.running_header_lines_removed += 1
            stats.remember(lines[second_idx])
            if reason == "title_like_after_page_marker":
                stats.warnings.add("possible_body_text_removed")
            return second_idx + 1
        stats.warnings.add("unusual_header_pattern")
        return start_idx

    is_header, reason = likely_running_header(first, article_title, after_page_marker=True)
    if is_header:
        stats.running_header_lines_removed += 1
        stats.remember(lines[first_idx])
        if reason == "title_like_after_page_marker":
            stats.warnings.add("possible_body_text_removed")
        next_idx = first_idx + 1
        while next_idx < len(lines) and not lines[next_idx].strip():
            next_idx += 1
        if next_idx < len(lines) and STANDALONE_PAGE_NUMBER_RE.match(lines[next_idx].strip()):
            standalone = lines[next_idx].strip()
            if not marker_page or standalone == marker_page or len(standalone) <= 4:
                stats.standalone_page_numbers_removed += 1
                stats.remember(lines[next_idx])
                return next_idx + 1
        return first_idx + 1

    stats.warnings.add("unusual_header_pattern")
    return start_idx


def remove_page_headers(raw_text: str, article_title: str | None = None) -> tuple[str, HeaderCleaningStats]:
    stats = HeaderCleaningStats()
    text = normalize_unicode(raw_text).replace("\f", "\n\n")
    lines = text.splitlines()
    lines = remove_initial_journal_front_matter(lines, stats)
    output_lines: list[str] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if PAGE_MARKER_RE.match(line):
            stats.generated_page_markers_removed += 1
            stats.remember(line)
            idx = consume_header_after_page_marker(lines, idx + 1, line, article_title, stats)
            continue
        output_lines.append(line)
        idx += 1
    if stats.generated_page_markers_removed == 0:
        stats.warnings.add("no_page_markers_found")
    if stats.running_header_lines_removed >= 25:
        stats.warnings.add("many_headers_removed")
    return "\n".join(output_lines), stats


def clean_readable_text_with_report(raw_text: str, article_title: str | None = None) -> tuple[str, HeaderCleaningStats]:
    text, stats = remove_page_headers(raw_text, article_title)
    text = re.sub(r"(?m)^\s*\[\s*\d+\s*\]\s*$", "\n", text)
    text = re.sub(r"(?m)^[-_#=*]{5,}\s*$", "\n", text)
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    paragraphs = []
    for para in re.split(r"\n\s*\n+", text):
        lines = [line.strip() for line in para.splitlines() if line.strip()]
        if not lines:
            continue
        joined = []
        for line in lines:
            if joined and should_join_line(joined[-1], line):
                joined[-1] = joined[-1].rstrip() + " " + line.lstrip()
            else:
                joined.append(line)
        paragraphs.append("\n".join(joined))
    return "\n\n".join(paragraphs).strip(), stats


def clean_readable_text(raw_text: str, article_title: str | None = None) -> str:
    cleaned, _stats = clean_readable_text_with_report(raw_text, article_title)
    return cleaned


def normalized_test_text(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", normalize_unicode(text).strip())


def run_header_cleaning_tests() -> None:
    cases = [
        {
            "name": "trailing-page-number-header",
            "title": "And the Adjacent Countries in the Second Century B.C.",
            "input": """## p. 5 (#213) ##############################################

AND THE ADJACENT COUNTRIES IN THE SECOND CENTURY B.C. 5

The body paragraph begins here.
""",
            "expected": "The body paragraph begins here.",
        },
        {
            "name": "standalone-page-number-after-header",
            "title": "And the Adjacent Countries in the Second Century B.C.",
            "input": """## p. 6 (#214) ##############################################

AND THE ADJACENT COUNTRIES IN THE SECOND CENTURY B.C.
6

Another body paragraph begins here.
""",
            "expected": "Another body paragraph begins here.",
        },
        {
            "name": "standalone-page-number-before-header",
            "title": "Early European researches into the flora of China",
            "input": """## p. 6 (#62) ###############################################

6
EARLY EUROPEAN RESEARCHES

Body after number-first header.
""",
            "expected": "Body after number-first header.",
        },
        {
            "name": "header-before-standalone-page-number",
            "title": "Early European researches into the flora of China",
            "input": """## p. 6 (#62) ###############################################

EARLY EUROPEAN RESEARCHES
6

Body after header-first page number.
""",
            "expected": "Body after header-first page number.",
        },
        {
            "name": "leading-page-number-same-line-header",
            "title": "Early European researches into the flora of China",
            "input": """## p. 6 (#62) ###############################################

6 EARLY EUROPEAN RESEARCHES

Body after leading-number header.
""",
            "expected": "Body after leading-number header.",
        },
        {
            "name": "preserve-article-marker-after-page-marker",
            "title": "Retrospect of events in China",
            "input": """## p. 181 (#203) ############################################

ARTICLE IX.
RETROSPECT OF EVENTS IN CHINA, 1873.
By Author Name

Opening body remains here.
""",
            "expected": """ARTICLE IX.
RETROSPECT OF EVENTS IN CHINA, 1873.
By Author Name

Opening body remains here.""",
        },
        {
            "name": "front-matter-before-article-one",
            "title": "Title of Article",
            "input": """JOURNAL
OF THE
NORTH-CHINA BRANCH
OF THE
ROYAL ASIATIC SOCIETY.

ARTICLE I.

Title of Article

By Author Name

Body begins here.
""",
            "expected": """ARTICLE I.

Title of Article

By Author Name

Body begins here.""",
        },
    ]
    results = []
    for case in cases:
        cleaned, stats = clean_readable_text_with_report(case["input"], case["title"])
        passed = normalized_test_text(cleaned) == normalized_test_text(case["expected"])
        results.append({
            "name": case["name"],
            "passed": passed,
            "generated_page_markers_removed": stats.generated_page_markers_removed,
            "running_header_lines_removed": stats.running_header_lines_removed,
            "standalone_page_numbers_removed": stats.standalone_page_numbers_removed,
            "journal_front_matter_removed": stats.journal_front_matter_removed,
            "warnings": sorted(stats.warnings or []),
            "cleaned": cleaned,
        })
    failed = [result for result in results if not result["passed"]]
    print(json.dumps({"header_cleaning_tests": results}, indent=2, ensure_ascii=False))
    if failed:
        raise SystemExit(1)


def should_join_line(previous: str, current: str) -> bool:
    if not previous or not current:
        return False
    if re.search(r"[.!?。！？;；:]$", previous.strip()):
        return False
    if len(previous.strip()) < 25:
        return False
    if re.match(r"^[A-Z][A-Z\s'.-]{4,}$", current.strip()):
        return False
    return True


def load_custom_stopwords(path: Path) -> set[str]:
    stopwords = set()
    if not path.exists():
        return stopwords
    text, _encoding = read_text_lossy(path)
    for line in text.splitlines():
        item = normalize_for_match(line)
        if item:
            stopwords.add(item)
            stopwords.update(tok for tok in item.split() if len(tok) > 1)
    return stopwords


def make_analysis_text(cleaned_text: str, lang: str, custom_stopwords: set[str]) -> tuple[str, int, int, str]:
    if lang == "zh-Hant":
        text = normalize_unicode(cleaned_text)
        text = re.sub(r"[^\u3400-\u9fffA-Za-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text, len(re.findall(r"[\u3400-\u9fff]|[A-Za-z0-9]+", text)), len(text), "zh_hant_conservative_character_preserving"
    if lang == "fr":
        tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", normalize_unicode(cleaned_text).lower())
        tokens = [tok for tok in tokens if len(tok) >= 2 and tok not in FR_STOPWORDS]
        text = " ".join(tokens)
        return text, len(tokens), len(text), "fr_lowercase_stopword_filter"
    if lang == "de":
        tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", normalize_unicode(cleaned_text).lower())
        tokens = [tok for tok in tokens if len(tok) >= 2 and tok not in DE_STOPWORDS]
        text = " ".join(tokens)
        return text, len(tokens), len(text), "de_lowercase_stopword_filter"
    if lang == "en":
        norm = normalize_for_match(cleaned_text)
        stopwords = EN_STOPWORDS | custom_stopwords
        tokens = [tok for tok in norm.split() if len(tok) >= 2 and not tok.isdigit() and tok not in stopwords]
        text = " ".join(tokens)
        return text, len(tokens), len(text), "en_lowercase_stopword_filter_custom_stopwords"
    text = re.sub(r"\s+", " ", normalize_unicode(cleaned_text)).strip()
    tokens = re.findall(r"[\wÀ-ÖØ-öø-ÿ\u3400-\u9fff]+", text, flags=re.UNICODE)
    return text, len(tokens), len(text), "unknown_or_mixed_conservative_normalization"


def split_paragraphs(text: str) -> list[tuple[str, int, int]]:
    paragraphs = []
    for match in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", text, flags=re.S):
        para = match.group(0).strip()
        if para:
            paragraphs.append((para, match.start(), match.end()))
    return paragraphs


def split_sentences(paragraph: str, lang: str) -> tuple[list[tuple[str, int, int]], str, str]:
    if lang == "zh-Hant":
        parts = list(re.finditer(r"[^。！？；]+[。！？；]?", paragraph))
        sentences = [(m.group(0).strip(), m.start(), m.end()) for m in parts if m.group(0).strip()]
        return sentences, "chinese_punctuation_regex", ""
    if lang in {"en", "fr", "de"}:
        parts = list(re.finditer(r"[^.!?]+(?:[.!?]+[\"')\]]*)?", paragraph))
        sentences = [(m.group(0).strip(), m.start(), m.end()) for m in parts if m.group(0).strip()]
        return sentences, f"{lang}_punctuation_regex", ""
    return [], "paragraph_only", "sentence segmentation skipped for unknown or mixed language"


def token_count_for_unit(text: str, lang: str) -> int:
    if lang == "zh-Hant":
        return len(re.findall(r"[\u3400-\u9fff]|[A-Za-z0-9]+", text))
    return len(re.findall(r"[\wÀ-ÖØ-öø-ÿ]+", text, flags=re.UNICODE))


def load_text_records(article_dir: Path) -> list[TextRecord]:
    records = []
    for idx, path in enumerate(sorted(article_dir.glob("*.txt")), start=1):
        raw_text, enc = read_text_lossy(path)
        year_hint, year_start, year_end = extract_year_hint(path.name)
        suffix = filename_suffix(path.name)
        suggested, final, source, exclude, reason, notes = infer_text_type(path.name)
        head_for_match = raw_text[:8000]
        head_norm = normalize_for_match(head_for_match)
        head_lines_norm = [normalize_for_match(line) for line in head_for_match.splitlines()[:45]]
        head_lines_norm = [line for line in head_lines_norm if line]
        lang, confidence, method, mixed, non_english, chinese = detect_language(raw_text)
        norm_raw = normalize_for_duplicate(raw_text)
        records.append(
            TextRecord(
                text_id=f"TXT_{idx:05d}",
                text_file=path.name,
                raw_text_path=path,
                raw_text=raw_text,
                encoding=enc,
                file_size=path.stat().st_size,
                raw_char_count=len(raw_text),
                raw_line_count=raw_text.count("\n") + 1 if raw_text else 0,
                raw_text_checksum=sha1_text(raw_text),
                raw_norm_checksum=sha1_text(norm_raw),
                filename_year_hint=year_hint,
                year_hint_start=year_start,
                year_hint_end=year_end,
                filename_suffix=suffix,
                suggested_text_type=suggested,
                final_text_type=final,
                text_type_source=source,
                exclude_from_article_analysis=exclude,
                exclusion_reason=reason,
                non_article_notes=notes,
                head_for_match=head_for_match,
                head_norm=head_norm,
                head_tokens=set(head_norm.split()),
                head_lines_norm=head_lines_norm,
                detected_language=lang,
                language_confidence=confidence,
                language_detection_method=method,
                possible_mixed_language_flag=mixed,
                non_english_flag=non_english,
                chinese_flag=chinese,
            )
        )
    return records


def collapse_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metadata = metadata.copy()
    metadata["master_id"] = metadata["master_id"].astype(str)
    base_cols = [
        "master_id", "year", "volume", "author", "author_raw", "title",
        "year_start", "year_end", "time_window", "notes",
    ]
    base_cols = [c for c in base_cols if c in metadata.columns]
    subject_cols = [
        "subject", "subject_l1", "subject_l2", "subject_raw",
        "subject_raw_norm", "subject_split", "mapping_note",
    ]
    subject_cols = [c for c in subject_cols if c in metadata.columns]
    weight_col = "subject_weight" if "subject_weight" in metadata.columns else None

    for master_id, group in metadata.groupby("master_id", sort=False):
        first = group.iloc[0]
        row = {col: first.get(col, "") for col in base_cols}
        row["metadata_row_count"] = len(group)
        for col in subject_cols:
            values = [str(v).strip() for v in group[col].dropna().tolist() if str(v).strip()]
            row[f"{col}_list"] = " | ".join(dict.fromkeys(values))
        if weight_col:
            weighted = []
            for _, item in group.iterrows():
                subject = str(item.get("subject", "")).strip()
                weight = item.get(weight_col, "")
                if subject:
                    weighted.append(f"{subject}:{weight}")
            row["subject_weight_list"] = " | ".join(weighted)
        rows.append(row)

    article_meta = pd.DataFrame(rows)
    article_meta["title_match_norm"] = article_meta["title"].map(normalize_for_match)
    article_meta["title_match_tokens"] = article_meta["title_match_norm"].map(
        lambda value: [tok for tok in value.split() if len(tok) > 2]
    )
    return article_meta


def cheap_title_score(title_norm: str, title_tokens: list[str], record: TextRecord) -> tuple[float, str]:
    if not title_norm:
        return 0.0, "missing_title"
    if title_norm in record.head_norm:
        return 100.0, "title_substring_in_head"
    if not title_tokens:
        return 0.0, "missing_title_tokens"
    coverage = sum(1 for tok in title_tokens if tok in record.head_tokens) / len(title_tokens)
    return round(coverage * 75.0, 2), f"coverage={coverage:.2f};preselect"


def title_score(title_norm: str, title_tokens: list[str], record: TextRecord) -> tuple[float, str]:
    if not title_norm:
        return 0.0, "missing_title"
    if title_norm in record.head_norm:
        return 100.0, "title_substring_in_head"
    if not title_tokens:
        return 0.0, "missing_title_tokens"
    coverage = sum(1 for tok in title_tokens if tok in record.head_tokens) / len(title_tokens)
    best_local = 0.0
    for idx in range(len(record.head_lines_norm)):
        for span in (1, 2):
            chunk_tokens = set(" ".join(record.head_lines_norm[idx:idx + span]).split())
            if chunk_tokens:
                best_local = max(best_local, sum(1 for tok in title_tokens if tok in chunk_tokens) / len(title_tokens))
    score = 92.0 if best_local >= 0.99 else max(coverage * 75.0, best_local * 85.0)
    return round(score, 2), f"coverage={coverage:.2f};local_coverage={best_local:.2f}"


def year_bonus(meta_row: pd.Series | dict, record: TextRecord) -> int:
    if record.year_hint_start is None:
        return 0
    meta_start = int(float(meta_row.get("year_start", 0) or 0))
    meta_end = int(float(meta_row.get("year_end", meta_start) or meta_start))
    if record.year_hint_start <= meta_end and (record.year_hint_end or record.year_hint_start) >= meta_start:
        return 12
    return -12


def confidence_label(score: float, margin: float, duplicate_best: bool, y_bonus: int) -> str:
    if y_bonus < 0:
        return "review_year_conflict" if score >= 70 else "unmatched"
    if duplicate_best and score < 100:
        return "review_tied_best"
    if score >= 92 and margin >= 5:
        return "high"
    if score >= 84 and margin >= 4:
        return "medium"
    if score >= 70:
        return "review"
    return "unmatched"


def build_match_reports(records: list[TextRecord], article_meta: pd.DataFrame, top_n: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_rows = []
    best_rows = []
    meta_records = list(article_meta.to_dict("records"))
    record_by_file = {record.text_file: record for record in records}

    for record in records:
        preselected = []
        for meta in meta_records:
            cheap_score, cheap_reason = cheap_title_score(meta["title_match_norm"], meta["title_match_tokens"], record)
            y_bonus = year_bonus(meta, record)
            preselected.append((cheap_score + y_bonus, cheap_score, y_bonus, meta, cheap_reason))
        preselected.sort(key=lambda item: item[0], reverse=True)

        scored = []
        for _combined, _cheap_score, y_bonus, meta, _cheap_reason in preselected[:18]:
            score, reason = title_score(meta["title_match_norm"], meta["title_match_tokens"], record)
            scored.append((score + y_bonus, score, y_bonus, meta, reason))
        scored.sort(key=lambda item: item[0], reverse=True)
        top = scored[:top_n]
        best_score = top[0][1] if top else 0.0
        second_score = top[1][1] if len(top) > 1 else 0.0
        margin = best_score - second_score
        duplicate_best = len(top) > 1 and abs(top[0][0] - top[1][0]) < 0.01

        for rank, (combined, raw_score, y_bonus, meta, reason) in enumerate(top, start=1):
            candidate_rows.append({
                "text_id": record.text_id,
                "text_file": record.text_file,
                "rank": rank,
                "master_id": meta["master_id"],
                "metadata_title": meta["title"],
                "metadata_year": meta.get("year", ""),
                "metadata_author": meta.get("author", ""),
                "title_score": raw_score,
                "year_bonus": y_bonus,
                "combined_score": round(combined, 2),
                "score_reason": reason,
                "detected_language": record.detected_language,
                "final_text_type": record.final_text_type,
                "exclude_from_article_analysis": record.exclude_from_article_analysis,
                "cleaned_readable_excerpt": "",
                "match_confidence": "",
                "year_conflict_flag": y_bonus < 0,
            })

        if top:
            combined, raw_score, y_bonus, meta, reason = top[0]
            label = confidence_label(raw_score, margin, duplicate_best, y_bonus)
            best_rows.append({
                "text_id": record.text_id,
                "text_file": record.text_file,
                "matched_master_id": meta["master_id"] if label != "unmatched" else "",
                "matched_title": meta["title"] if label != "unmatched" else "",
                "matched_year": meta.get("year", "") if label != "unmatched" else "",
                "matched_author": meta.get("author", "") if label != "unmatched" else "",
                "title_score": raw_score,
                "second_title_score": second_score,
                "score_margin": round(margin, 2),
                "year_bonus": y_bonus,
                "combined_score": round(combined, 2),
                "match_confidence": label,
                "score_reason": reason,
                "year_conflict_flag": y_bonus < 0,
                "detected_language": record.detected_language,
                "final_text_type": record.final_text_type,
                "exclude_from_article_analysis": record.exclude_from_article_analysis,
                "cleaned_readable_excerpt": "",
            })

    return pd.DataFrame(candidate_rows), pd.DataFrame(best_rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def safe_to_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_new{path.suffix}")
        df.to_csv(fallback, index=False, encoding="utf-8-sig")
        return fallback


def manifest_rows(records: list[TextRecord]) -> list[dict]:
    rows = []
    for record in records:
        rows.append({
            "text_id": record.text_id,
            "text_file": record.text_file,
            "raw_text_path": str(record.raw_text_path),
            "file_size": record.file_size,
            "encoding": record.encoding,
            "raw_text_checksum": record.raw_text_checksum,
            "raw_char_count": record.raw_char_count,
            "raw_line_count": record.raw_line_count,
            "filename_year_hint": record.filename_year_hint,
            "filename_suffix": record.filename_suffix,
            "possible_non_article_flag": record.suggested_text_type != "article_candidate",
            "suggested_text_type": record.suggested_text_type,
            "notes": record.non_article_notes,
            "detected_language": record.detected_language,
            "language_confidence": record.language_confidence,
            "language_detection_method": record.language_detection_method,
            "possible_mixed_language_flag": record.possible_mixed_language_flag,
            "non_english_flag": record.non_english_flag,
            "chinese_flag": record.chinese_flag,
            "final_text_type": record.final_text_type,
            "text_type_source": record.text_type_source,
            "exclude_from_article_analysis": record.exclude_from_article_analysis,
            "exclusion_reason": record.exclusion_reason,
        })
    return rows


def build_text_units(text_outputs: pd.DataFrame, records_by_file: dict[str, TextRecord], output_path: Path) -> pd.DataFrame:
    unit_rows = []
    diagnostics = []
    for _, row in text_outputs.iterrows():
        record = records_by_file[row["text_file"]]
        text = Path(row["cleaned_readable_text_path"]).read_text(encoding="utf-8")
        paragraph_count = 0
        sentence_count = 0
        methods = set()
        warnings = set()
        unit_num = 0
        for para_idx, (para, p_start, p_end) in enumerate(split_paragraphs(text), start=1):
            paragraph_count += 1
            unit_num += 1
            unit_rows.append({
                "text_file": record.text_file,
                "unit_id": f"{record.text_id}_P{para_idx:04d}",
                "unit_type": "paragraph",
                "unit_order": unit_num,
                "paragraph_id": para_idx,
                "sentence_id": "",
                "unit_text": para,
                "detected_language": record.detected_language,
                "char_start": p_start,
                "char_end": p_end,
                "token_count": token_count_for_unit(para, record.detected_language),
                "source_cleaned_readable_path": row["cleaned_readable_text_path"],
            })
            sentences, method, warning = split_sentences(para, record.detected_language)
            methods.add(method)
            if warning:
                warnings.add(warning)
            for sent_idx, (sent, s_start, s_end) in enumerate(sentences, start=1):
                sentence_count += 1
                unit_num += 1
                unit_rows.append({
                    "text_file": record.text_file,
                    "unit_id": f"{record.text_id}_P{para_idx:04d}_S{sent_idx:04d}",
                    "unit_type": "sentence",
                    "unit_order": unit_num,
                    "paragraph_id": para_idx,
                    "sentence_id": sent_idx,
                    "unit_text": sent,
                    "detected_language": record.detected_language,
                    "char_start": p_start + s_start,
                    "char_end": p_start + s_end,
                    "token_count": token_count_for_unit(sent, record.detected_language),
                    "source_cleaned_readable_path": row["cleaned_readable_text_path"],
                })
        diagnostics.append({
            "text_file": record.text_file,
            "paragraph_count": paragraph_count,
            "sentence_count": sentence_count,
            "sentence_segmentation_method": ";".join(sorted(methods)) if methods else "none",
            "sentence_segmentation_warning": ";".join(sorted(warnings)),
        })
    units = pd.DataFrame(unit_rows)
    safe_to_csv(units, output_path)
    return pd.DataFrame(diagnostics)


def duplicate_groups(master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    dup_source = master[master["analysis_char_count"].fillna(0).astype(int) > 0].copy()
    by_hash = defaultdict(list)
    for _, row in dup_source.iterrows():
        by_hash[row["duplicate_hash"]].append(row)
    group_num = 0
    for _hash, group in by_hash.items():
        if len(group) < 2:
            continue
        group_num += 1
        keep_file = sorted(group, key=lambda r: (
            r.get("exclude_from_article_analysis", True),
            0 if r.get("match_confidence") in {"high", "medium"} else 1,
            -int(r.get("analysis_token_count") or 0),
            r.get("text_file", ""),
        ))[0]["text_file"]
        for row in group:
            rows.append({
                "duplicate_group_id": f"DUP_{group_num:04d}",
                "text_file": row["text_file"],
                "final_text_type": row.get("final_text_type", ""),
                "exclude_from_article_analysis": row.get("exclude_from_article_analysis", ""),
                "detected_language": row.get("detected_language", ""),
                "matched_master_id": row.get("matched_master_id", ""),
                "matched_title": row.get("matched_title", ""),
                "matched_year": row.get("matched_year", ""),
                "match_confidence": row.get("match_confidence", ""),
                "cleaned_readable_excerpt": row.get("cleaned_readable_excerpt", ""),
                "suggested_keep_or_exclude": "keep" if row["text_file"] == keep_file else "exclude_duplicate",
            })
    return pd.DataFrame(rows)


def truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def add_final_dataset_flags(master: pd.DataFrame, duplicate_df: pd.DataFrame) -> pd.DataFrame:
    master = master.copy()
    if duplicate_df.empty:
        duplicate_lookup = {}
        keep_lookup = {}
    else:
        duplicate_lookup = duplicate_df.set_index("text_file")["duplicate_group_id"].to_dict()
        keep_lookup = duplicate_df.set_index("text_file")["suggested_keep_or_exclude"].to_dict()
    master["duplicate_group_id"] = master["text_file"].map(duplicate_lookup).fillna("")
    master["duplicate_flag"] = master["duplicate_group_id"].astype(str) != ""
    master["suggested_keep_or_exclude"] = master["text_file"].map(keep_lookup).fillna("keep")
    master["needs_match_review_flag"] = ~master["match_confidence"].isin(["high", "medium"])
    reasons = []
    include = []
    for _, row in master.iterrows():
        row_reasons = []
        if truthy(row.get("exclude_from_article_analysis", False)):
            row_reasons.append(row.get("exclusion_reason") or "excluded_text_type")
        if row.get("suggested_keep_or_exclude") == "exclude_duplicate":
            row_reasons.append("duplicate_copy_not_selected")
        if not str(row.get("cleaned_readable_text_path", "")).strip():
            row_reasons.append("missing_cleaned_readable_text")
        if not str(row.get("analysis_text_path", "")).strip():
            row_reasons.append("missing_analysis_text")
        if row.get("match_confidence") not in {"high", "medium"}:
            row_reasons.append("metadata_match_needs_review")
        include.append(not row_reasons)
        reasons.append("; ".join(dict.fromkeys(str(r) for r in row_reasons if str(r).strip())))
    master["include_in_final_article_analysis"] = include
    master["final_dataset_exclusion_reason"] = reasons
    return master


def safe_mkdirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def create_language_distribution(master: pd.DataFrame, output_path: Path) -> None:
    df = master.copy()
    df["article_vs_non_article_status"] = df["exclude_from_article_analysis"].map(lambda v: "excluded_non_article_or_uncertain" if truthy(v) else "article_candidate")
    df["year_or_decade"] = df["filename_year_hint"].fillna("").map(lambda v: f"{int(str(v)[:4]) // 10 * 10}s" if re.match(r"^\d{4}", str(v)) else "")
    grouped = df.groupby(
        ["detected_language", "final_text_type", "article_vs_non_article_status", "year_or_decade"],
        dropna=False,
    ).size().reset_index(name="count")
    safe_to_csv(grouped, output_path)


def series_counts(df: pd.DataFrame, column: str, top: int | None = None) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype=int)
    counts = df[column].fillna("").astype(str).replace("", "missing").value_counts()
    return counts.head(top) if top else counts


def describe_numeric(series: pd.Series) -> str:
    values = [float(v) for v in series.dropna().tolist() if str(v) != "" and not pd.isna(v)]
    if not values:
        return "no values"
    return f"min={min(values):.0f}, median={statistics.median(values):.0f}, mean={statistics.mean(values):.1f}, max={max(values):.0f}"


def write_markdown_table(counts: pd.Series) -> list[str]:
    lines = ["| value | count |", "|---|---:|"]
    for key, value in counts.items():
        lines.append(f"| {key} | {int(value)} |")
    return lines


def make_plots(master: pd.DataFrame, figures_dir: Path) -> list[str]:
    made = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return make_pillow_plots(master, figures_dir)

    def save_bar(series: pd.Series, filename: str, title: str, xlabel: str = "") -> None:
        if series.empty:
            return
        fig, ax = plt.subplots(figsize=(10, 5))
        series.plot(kind="bar", ax=ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(figures_dir / filename, dpi=180)
        plt.close(fig)
        made.append(filename)

    df = master.copy()
    year_series = pd.to_numeric(df["matched_year"], errors="coerce").dropna().astype(int)
    if not year_series.empty:
        save_bar(year_series.value_counts().sort_index(), "article_count_by_year.png", "Article count by year", "year")
        save_bar((year_series // 10 * 10).value_counts().sort_index(), "article_count_by_decade.png", "Article count by decade", "decade")
    save_bar(series_counts(df, "detected_language"), "language_distribution.png", "Language distribution")
    save_bar(series_counts(df, "final_text_type"), "text_type_distribution.png", "Text type distribution")
    save_bar(series_counts(df, "match_confidence"), "match_confidence_distribution.png", "Match confidence distribution")

    lengths = pd.to_numeric(df["analysis_token_count"], errors="coerce").dropna()
    if not lengths.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(lengths, bins=30)
        ax.set_title("Text length distribution")
        ax.set_xlabel("analysis tokens")
        ax.set_ylabel("files")
        fig.tight_layout()
        fig.savefig(figures_dir / "text_length_distribution.png", dpi=180)
        plt.close(fig)
        made.append("text_length_distribution.png")

    subjects = Counter()
    for value in df.get("subject_list", pd.Series(dtype=str)).fillna(""):
        for item in str(value).split(" | "):
            if item.strip():
                subjects[item.strip()] += 1
    if subjects:
        save_bar(pd.Series(dict(subjects.most_common(15))), "top_subjects_overall.png", "Top subjects overall")
    if "subject_list" in df.columns and not year_series.empty:
        rows = []
        for _, row in df.iterrows():
            year = pd.to_numeric(pd.Series([row.get("matched_year")]), errors="coerce").iloc[0]
            if pd.isna(year):
                continue
            decade = int(year) // 10 * 10
            for subject in str(row.get("subject_list", "")).split(" | "):
                if subject.strip():
                    rows.append({"decade": decade, "subject": subject.strip()})
        if rows:
            top_by_decade = pd.DataFrame(rows).groupby(["decade", "subject"]).size().reset_index(name="count")
            top_subjects = top_by_decade.groupby("subject")["count"].sum().sort_values(ascending=False).head(8).index
            pivot = top_by_decade[top_by_decade["subject"].isin(top_subjects)].pivot(index="decade", columns="subject", values="count").fillna(0)
            if not pivot.empty:
                fig, ax = plt.subplots(figsize=(11, 6))
                pivot.plot(kind="bar", stacked=True, ax=ax)
                ax.set_title("Top subjects by decade")
                ax.set_xlabel("decade")
                ax.set_ylabel("count")
                fig.tight_layout()
                fig.savefig(figures_dir / "top_subjects_by_decade.png", dpi=180)
                plt.close(fig)
                made.append("top_subjects_by_decade.png")
    return made


def make_pillow_plots(master: pd.DataFrame, figures_dir: Path) -> list[str]:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return []

    made = []

    def font(size: int):
        try:
            return ImageFont.truetype("arial.ttf", size)
        except Exception:
            return ImageFont.load_default()

    def save_bar(series: pd.Series, filename: str, title: str, max_items: int = 30) -> None:
        series = series.dropna()
        if series.empty:
            return
        if len(series) > max_items:
            series = series.head(max_items)
        labels = [str(idx) for idx in series.index]
        values = [int(v) for v in series.values]
        width, height = 1200, 720
        margin_l, margin_r, margin_t, margin_b = 180, 40, 80, 180
        plot_w = width - margin_l - margin_r
        plot_h = height - margin_t - margin_b
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)
        title_font = font(26)
        label_font = font(14)
        small_font = font(12)
        draw.text((margin_l, 28), title, fill=(30, 30, 30), font=title_font)
        max_value = max(values) or 1
        bar_gap = 4
        bar_w = max(8, int((plot_w - bar_gap * (len(values) - 1)) / len(values)))
        axis_color = (90, 90, 90)
        draw.line((margin_l, margin_t, margin_l, margin_t + plot_h), fill=axis_color, width=2)
        draw.line((margin_l, margin_t + plot_h, width - margin_r, margin_t + plot_h), fill=axis_color, width=2)
        for i in range(5):
            y = margin_t + plot_h - int(plot_h * i / 4)
            value = int(max_value * i / 4)
            draw.line((margin_l - 5, y, width - margin_r, y), fill=(230, 230, 230), width=1)
            draw.text((20, y - 8), str(value), fill=(70, 70, 70), font=small_font)
        for idx, (label, value) in enumerate(zip(labels, values)):
            x0 = margin_l + idx * (bar_w + bar_gap)
            x1 = x0 + bar_w
            bar_h = int(plot_h * value / max_value)
            y0 = margin_t + plot_h - bar_h
            y1 = margin_t + plot_h
            color = (63, 112, 164) if idx % 2 == 0 else (88, 141, 105)
            draw.rectangle((x0, y0, x1, y1), fill=color)
            draw.text((x0, max(0, y0 - 18)), str(value), fill=(30, 30, 30), font=small_font)
            truncated = label[:18]
            tx = x0
            ty = margin_t + plot_h + 10
            draw.text((tx, ty), truncated, fill=(30, 30, 30), font=label_font)
        img.save(figures_dir / filename)
        made.append(filename)

    df = master.copy()
    years = pd.to_numeric(df["matched_year"], errors="coerce").dropna().astype(int)
    if not years.empty:
        save_bar(years.value_counts().sort_index(), "article_count_by_year.png", "Article count by year", 60)
        save_bar((years // 10 * 10).value_counts().sort_index(), "article_count_by_decade.png", "Article count by decade")
    save_bar(series_counts(df, "detected_language"), "language_distribution.png", "Language distribution")
    save_bar(series_counts(df, "final_text_type"), "text_type_distribution.png", "Text type distribution")
    save_bar(series_counts(df, "match_confidence"), "match_confidence_distribution.png", "Match confidence distribution")
    lengths = pd.to_numeric(df["analysis_token_count"], errors="coerce").dropna()
    if not lengths.empty:
        bins = pd.cut(lengths, bins=12).value_counts().sort_index()
        bins.index = [f"{int(interval.left)}-{int(interval.right)}" for interval in bins.index]
        save_bar(bins, "text_length_distribution.png", "Text length distribution", 12)
    subjects = Counter()
    for value in df.get("subject_list", pd.Series(dtype=str)).fillna(""):
        for item in str(value).split(" | "):
            if item.strip():
                subjects[item.strip()] += 1
    if subjects:
        save_bar(pd.Series(dict(subjects.most_common(15))), "top_subjects_overall.png", "Top subjects overall", 15)
    rows = []
    for _, row in df.iterrows():
        year = pd.to_numeric(pd.Series([row.get("matched_year")]), errors="coerce").iloc[0]
        if pd.isna(year):
            continue
        decade = int(year) // 10 * 10
        for subject in str(row.get("subject_list", "")).split(" | "):
            if subject.strip():
                rows.append({"decade": decade, "subject": subject.strip()})
    if rows:
        top_by_decade = pd.DataFrame(rows).groupby(["decade", "subject"]).size().reset_index(name="count")
        top_subjects = set(top_by_decade.groupby("subject")["count"].sum().sort_values(ascending=False).head(8).index)
        compact = top_by_decade[top_by_decade["subject"].isin(top_subjects)].groupby("decade")["count"].sum()
        save_bar(compact.sort_index(), "top_subjects_by_decade.png", "Top subject assignments by decade")
    return made


def write_corpus_diagnostics(master: pd.DataFrame, duplicate_df: pd.DataFrame, figures: list[str], output_path: Path) -> None:
    final_count = int(master["include_in_final_article_analysis"].sum())
    lines = [
        "# Corpus Diagnostics",
        "",
        f"- Total raw `.txt` files: {len(master)}",
        f"- Total article candidates: {int((master['suggested_text_type'] == 'article_candidate').sum())}",
        f"- Total excluded non-articles or uncertain sections: {int(master['exclude_from_article_analysis'].map(truthy).sum())}",
        f"- Total final article-analysis texts: {final_count}",
        "",
        "## Counts by detected language",
        *write_markdown_table(series_counts(master, "detected_language")),
        "",
        "## Counts by text type",
        *write_markdown_table(series_counts(master, "final_text_type")),
        "",
        "## Counts by year and decade",
    ]
    years = pd.to_numeric(master["matched_year"], errors="coerce").dropna().astype(int)
    lines.extend(write_markdown_table(years.value_counts().sort_index().tail(25)) if not years.empty else ["No matched years available."])
    lines.append("")
    lines.append("## Metadata match confidence distribution")
    lines.extend(write_markdown_table(series_counts(master, "match_confidence")))
    lines.extend([
        "",
        f"- Files needing match review: {int(master['needs_match_review_flag'].sum())}",
        f"- Duplicate groups: {int(duplicate_df['duplicate_group_id'].nunique()) if not duplicate_df.empty else 0}",
        f"- Text length distribution: {describe_numeric(master['analysis_token_count'])}",
        f"- Paragraph count distribution: {describe_numeric(master['paragraph_count'])}",
        f"- Sentence count distribution: {describe_numeric(master['sentence_count'])}",
        "",
        "## Top subjects overall",
    ])
    subjects = Counter()
    for value in master.get("subject_list", pd.Series(dtype=str)).fillna(""):
        for item in str(value).split(" | "):
            if item.strip():
                subjects[item.strip()] += 1
    lines.extend(write_markdown_table(pd.Series(dict(subjects.most_common(15)))) if subjects else ["No subject data available."])
    lines.extend([
        "",
        "## Top subjects by decade",
        "See `outputs/figures/top_subjects_by_decade.png` when enough matched year and subject data are available.",
        "",
        "## Figures",
    ])
    lines.extend([f"- `{name}`" for name in figures] or ["- Plot generation was skipped because plotting libraries were unavailable."])
    lines.extend([
        "",
        "## Multilingual preprocessing warnings",
        "- Language detection uses a local heuristic so uncertain and mixed-language files should be reviewed manually.",
        "- English, French, German, and Chinese analysis texts are prepared separately; final cross-language topic modeling should wait until language-specific QA is complete.",
        "- Traditional Chinese text is preserved and is not converted to Simplified Chinese.",
    ])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_analysis_ready(master: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    final = master[master["include_in_final_article_analysis"]].copy()
    if "matched_year" in final.columns:
        final["year"] = pd.to_numeric(final["matched_year"], errors="coerce")
        final["decade"] = final["year"].map(lambda v: int(v) // 10 * 10 if not pd.isna(v) else "")
    else:
        final["year"] = ""
        final["decade"] = ""
    columns = [
        "text_id", "text_file", "matched_master_id", "year", "decade", "matched_title",
        "subject_list", "cleaned_readable_text_path", "analysis_text_path",
        "detected_language", "analysis_token_count", "match_confidence",
    ]
    for lang, filename in [
        ("en", "en_articles.csv"),
        ("fr", "fr_articles.csv"),
        ("de", "de_articles.csv"),
        ("zh-Hant", "zh_hant_articles.csv"),
    ]:
        safe_to_csv(final[final["detected_language"] == lang][columns], output_dir / filename)
    safe_to_csv(
        final[~final["detected_language"].isin(["en", "fr", "de", "zh-Hant"])][columns],
        output_dir / "mixed_or_unknown_articles.csv",
    )
    en = final[final["detected_language"] == "en"].copy()
    en_input_cols = columns + ["analysis_text"]
    if not en.empty:
        en["analysis_text"] = en["analysis_text_path"].map(lambda p: Path(p).read_text(encoding="utf-8") if p and Path(p).exists() else "")
    else:
        en["analysis_text"] = ""
    safe_to_csv(en[en_input_cols], output_dir / "en_nmf_input.csv")
    tfidf = english_tfidf_by_decade(en)
    safe_to_csv(tfidf, output_dir / "en_tfidf_terms_by_decade.csv")
    return final, tfidf


def english_tfidf_by_decade(en: pd.DataFrame) -> pd.DataFrame:
    if en.empty or "decade" not in en.columns:
        return pd.DataFrame(columns=["decade", "term", "tfidf"])
    docs = []
    for decade, group in en.groupby("decade"):
        if decade == "":
            continue
        text = " ".join(group["analysis_text"].fillna("").astype(str).tolist())
        tokens = [tok for tok in text.split() if len(tok) > 2]
        if tokens:
            docs.append((decade, Counter(tokens)))
    if not docs:
        return pd.DataFrame(columns=["decade", "term", "tfidf"])
    df_counts = Counter()
    for _decade, counts in docs:
        for term in counts:
            df_counts[term] += 1
    rows = []
    doc_n = len(docs)
    for decade, counts in docs:
        total = sum(counts.values()) or 1
        scored = []
        for term, count in counts.items():
            tf = count / total
            idf = math.log((1 + doc_n) / (1 + df_counts[term])) + 1
            scored.append((term, tf * idf))
        for term, score in sorted(scored, key=lambda item: item[1], reverse=True)[:40]:
            rows.append({"decade": decade, "term": term, "tfidf": round(score, 8)})
    return pd.DataFrame(rows)


def write_preliminary_analysis_summary(final: pd.DataFrame, tfidf: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# Preliminary Analysis Summary",
        "",
        "This is exploratory only. Final topic modeling has not been run.",
        "",
        "## Corpus size by language",
        *write_markdown_table(series_counts(final, "detected_language")),
        "",
        "## Corpus size by decade",
        *write_markdown_table(series_counts(final, "decade")),
        "",
        "## Top metadata subjects",
    ]
    subjects = Counter()
    for value in final.get("subject_list", pd.Series(dtype=str)).fillna(""):
        for item in str(value).split(" | "):
            if item.strip():
                subjects[item.strip()] += 1
    lines.extend(write_markdown_table(pd.Series(dict(subjects.most_common(15)))) if subjects else ["No final matched subject data available."])
    lines.extend(["", "## Top frequent terms by language"])
    for lang in ["en", "fr", "de", "zh-Hant", "mixed", "unknown"]:
        subset = final[final["detected_language"] == lang]
        counts = Counter()
        for path in subset.get("analysis_text_path", pd.Series(dtype=str)).fillna(""):
            p = Path(path)
            if p.exists():
                counts.update(tok for tok in p.read_text(encoding="utf-8").split() if len(tok) > 1)
        if counts:
            lines.append(f"- {lang}: " + ", ".join(f"{term} ({count})" for term, count in counts.most_common(20)))
    lines.extend(["", "## English frequent terms by decade"])
    en = final[final["detected_language"] == "en"]
    for decade, group in en.groupby("decade"):
        counts = Counter()
        for path in group["analysis_text_path"].fillna(""):
            p = Path(path)
            if p.exists():
                counts.update(tok for tok in p.read_text(encoding="utf-8").split() if len(tok) > 2)
        if counts:
            lines.append(f"- {decade}: " + ", ".join(f"{term} ({count})" for term, count in counts.most_common(20)))
    lines.extend(["", "## English TF-IDF keywords by decade"])
    if tfidf.empty:
        lines.append("No English TF-IDF table was generated.")
    else:
        for decade, group in tfidf.groupby("decade"):
            lines.append(f"- {decade}: " + ", ".join(group.head(15)["term"].astype(str).tolist()))
    lines.extend([
        "",
        "## Notes",
        "- Non-English texts should be analyzed separately until language-specific preprocessing is manually reviewed.",
        "- OCR noise and inconsistent historical typography may affect frequency, TF-IDF, and downstream topic models.",
        "- Cross-language comparison should use metadata and carefully validated language-specific features rather than a single pooled bag of words.",
    ])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def excerpt(text: str, limit: int) -> str:
    return re.sub(r"\s+", " ", text).strip()[:limit]


def write_text_version_diagnostics(master: pd.DataFrame, output_path: Path, records_by_file: dict[str, TextRecord]) -> None:
    selected = []
    for lang in ["en", "fr", "de", "zh-Hant"]:
        subset = master[master["detected_language"] == lang]
        if not subset.empty:
            selected.append(subset.iloc[0])
    if len(selected) < 5:
        existing = {row["text_file"] for row in selected}
        for _, row in master.iterrows():
            if row["text_file"] not in existing:
                selected.append(row)
            if len(selected) >= 5:
                break
    lines = ["# Text Version Diagnostics", ""]
    for row in selected[:8]:
        record = records_by_file[row["text_file"]]
        cleaned = Path(row["cleaned_readable_text_path"]).read_text(encoding="utf-8") if Path(row["cleaned_readable_text_path"]).exists() else ""
        analysis = Path(row["analysis_text_path"]).read_text(encoding="utf-8") if Path(row["analysis_text_path"]).exists() else ""
        analysis_preview = " ".join(analysis.split()[:200])
        lines.extend([
            f"## {row['text_file']}",
            f"- Detected language: {row['detected_language']} ({row['language_confidence']})",
            f"- Final text type: {row['final_text_type']}",
            f"- Paragraph count: {row['paragraph_count']}",
            f"- Sentence count: {row['sentence_count']}",
            f"- Preprocessing profile: {row['analysis_preprocessing_profile']}",
            f"- Notes: {'review recommended' if row['detected_language'] in {'mixed', 'unknown'} or truthy(row['exclude_from_article_analysis']) else 'looks structurally usable'}",
            "",
            "Raw excerpt:",
            "",
            f"> {excerpt(record.raw_text, 500)}",
            "",
            "Cleaned readable excerpt:",
            "",
            f"> {excerpt(cleaned, 500)}",
            "",
            "Analysis text excerpt:",
            "",
            f"> {excerpt(analysis_preview, 1200)}",
            "",
        ])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def snippet_around_first_page_marker(raw_text: str, cleaned_text: str, removed_lines: str) -> tuple[str, str]:
    raw = normalize_unicode(raw_text)
    marker = PAGE_MARKER_RE.search(raw)
    if marker:
        start = max(0, marker.start() - 250)
        end = min(len(raw), marker.end() + 700)
        before = raw[start:end].strip()
    else:
        before = raw[:900].strip()
    removed_first = str(removed_lines).split(" | ")[0].strip()
    if removed_first and removed_first in cleaned_text:
        pos = cleaned_text.find(removed_first)
    else:
        pos = 0
    after = cleaned_text[max(0, pos - 100):pos + 900].strip() if pos else cleaned_text[:900].strip()
    return before, after


def write_header_cleaning_samples(
    records: list[TextRecord],
    master: pd.DataFrame,
    header_report: pd.DataFrame,
    output_path: Path,
) -> None:
    record_lookup = {record.text_file: record for record in records}
    master_lookup = master.set_index("text_file").to_dict("index") if not master.empty else {}
    selected: list[str] = []

    def add_files(subset: pd.DataFrame, limit: int) -> None:
        for filename in subset["filename"].tolist():
            if filename not in selected:
                selected.append(filename)
            if len(selected) >= limit:
                break

    add_files(header_report[header_report["generated_page_marker_lines_removed"] > 0], 5)
    add_files(
        header_report[
            header_report["first_10_removed_lines"].astype(str).str.contains(r"\b\d{1,4}\b", regex=True, na=False)
            & (header_report["running_header_lines_removed"] > 0)
        ],
        8,
    )
    add_files(header_report[header_report["standalone_page_number_lines_removed"] > 0], 10)
    add_files(header_report[header_report["journal_front_matter_removed"].map(truthy)], 12)
    add_files(header_report[header_report["warnings_or_uncertainty_flags"].astype(str).str.len() > 0], 15)

    lines = [
        "# Header Cleaning Samples",
        "",
        "These snippets show raw text around likely page headers and the corresponding cleaned v2 text.",
        "",
    ]
    for filename in selected[:15]:
        record = record_lookup.get(filename)
        master_row = master_lookup.get(filename, {})
        report_row = header_report[header_report["filename"] == filename].iloc[0]
        cleaned_path = Path(str(master_row.get("cleaned_readable_text_path", "")))
        cleaned = cleaned_path.read_text(encoding="utf-8") if cleaned_path.exists() else ""
        before, after = snippet_around_first_page_marker(
            record.raw_text if record else "",
            cleaned,
            str(report_row.get("first_10_removed_lines", "")),
        )
        lines.extend([
            f"## {filename}",
            f"- Matched title: {report_row.get('matched_title', '')}",
            f"- Removed page markers: {report_row.get('generated_page_marker_lines_removed', 0)}",
            f"- Removed running headers: {report_row.get('running_header_lines_removed', 0)}",
            f"- Removed standalone page numbers: {report_row.get('standalone_page_number_lines_removed', 0)}",
            f"- Front matter removed: {report_row.get('journal_front_matter_removed', False)}",
            f"- Warnings: {report_row.get('warnings_or_uncertainty_flags', '')}",
            "",
            "Before:",
            "",
            "```txt",
            before[:1500],
            "```",
            "",
            "After:",
            "",
            "```txt",
            after[:1500],
            "```",
            "",
        ])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_transformation_log(project_dir: Path, output_dir: Path, summary: dict, run_label: str = RUN_LABEL) -> None:
    lines = [
        "# RAS Text Dataset Pipeline Log",
        "",
        "## Inputs and preservation",
        f"- Project folder: `{project_dir}`",
        "- Raw files in `articles/` are preserved and are never modified, deleted, or overwritten.",
        "- Every raw `.txt` file is listed in `outputs/raw_text_manifest.csv`.",
        "",
        "## Non-article handling",
        "- Non-articles are excluded by filename-derived flags only.",
        "- Clear suffixes such as review, index, proceedings, minutes, bibliography, front matter, appendix, and advertisement set `exclude_from_article_analysis = TRUE`.",
        "- Uncertain suffixes such as notes or miscellaneous sections are conservatively excluded for manual review.",
        "",
        "## Cleaned readable text",
        f"- Written to `outputs/{READABLE_OUTPUT_DIR_NAME}/`.",
        "- Preserves capitalization, punctuation, paragraph breaks, historical wording, diacritics, and Chinese characters.",
        "- Applies conservative OCR whitespace cleanup, generated page-marker removal, page-header removal, and hyphenated line-break repair.",
        f"- Opening title/author information is preserved by default: `KEEP_OPENING_TITLE_AUTHOR = {KEEP_OPENING_TITLE_AUTHOR}`.",
        "",
        "## Analysis text",
        f"- Written to `outputs/{WORDBAG_OUTPUT_DIR_NAME}/` from the cleaned readable version.",
        "- English receives lowercasing, English stopword filtering, and custom stopwords from `hit_stopwords.txt`.",
        "- French and German receive language-specific stopword filtering while preserving accents and umlauts.",
        "- Traditional Chinese is preserved with conservative character-aware normalization; it is not converted to Simplified Chinese.",
        "- Mixed or unknown-language texts receive conservative whitespace normalization only.",
        "",
        "## Language detection",
        "- Uses a local heuristic based on CJK script detection, Latin stopword evidence, and French/German diacritics.",
        "- Mixed and uncertain files are flagged for review.",
        "",
        "## Text units",
        "- Paragraph and sentence units are segmented from cleaned readable texts.",
        "- Chinese sentence splitting uses Chinese punctuation; unknown and mixed texts receive paragraph-level units only.",
        "",
        "## Metadata matching",
        "- Metadata is collapsed to one row per `master_id`.",
        "- Titles are matched against readable raw/cleaned header excerpts, not bag-of-words analysis text.",
        "- Filename years are used as a bonus or conflict warning.",
        "",
        "## Duplicate detection",
        "- Exact duplicates are detected from normalized readable text hashes.",
        "- Duplicate raw files are not deleted; reports suggest which copy to keep for analysis.",
        "",
        "## Final dataset rules",
        "- Final article-analysis rows must be article candidates, non-duplicate keepers, have both text versions, and have high or medium metadata-match confidence.",
        "- Excluded rows and reasons are written to `outputs/reports/final_dataset_exclusion_report.csv`.",
        "",
        "## Known limitations",
        "- Language detection is heuristic, not a trained multilingual classifier.",
        "- Chinese segmentation is conservative if no dedicated tokenizer is installed.",
        "- Topic models are intentionally not fitted in this pipeline; only analysis-ready files and exploratory diagnostics are prepared.",
        "",
        "## Run summary",
        f"- Metadata rows: {summary['metadata_rows']}",
        f"- Metadata article records: {summary['metadata_unique_articles']}",
        f"- Text files: {summary['text_files']}",
        f"- Matches by confidence: {summary['matches_by_confidence']}",
        f"- Exact duplicate groups: {summary['exact_duplicate_groups']}",
        f"- Final article-analysis texts: {summary['final_article_analysis_texts']}",
        f"- Header cleaning summary: {summary.get('header_cleaning', {})}",
    ]
    (output_dir / "reports" / f"transformation_log_{run_label}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_dataset(project_dir: Path) -> dict:
    article_dir = project_dir / "articles"
    metadata_path = project_dir / "ras_subject_expanded.csv"
    stopword_path = project_dir / "hit_stopwords.txt"
    output_dir = project_dir / "outputs"
    cleaned_dir = output_dir / READABLE_OUTPUT_DIR_NAME
    analysis_dir = output_dir / WORDBAG_OUTPUT_DIR_NAME
    text_units_dir = output_dir / f"text_units_{RUN_LABEL}"
    report_dir = output_dir / "reports"
    figures_dir = output_dir / f"figures_{RUN_LABEL}"
    analysis_ready_dir = output_dir / f"analysis_ready_{RUN_LABEL}"
    safe_mkdirs(cleaned_dir, analysis_dir, text_units_dir, report_dir, figures_dir, analysis_ready_dir)

    metadata = pd.read_csv(metadata_path, encoding="utf-8-sig", dtype={"master_id": str})
    article_meta = collapse_metadata(metadata)
    records = load_text_records(article_dir)
    records_by_file = {record.text_file: record for record in records}
    custom_stopwords = load_custom_stopwords(stopword_path)

    manifest = pd.DataFrame(manifest_rows(records))
    safe_to_csv(manifest, output_dir / f"raw_text_manifest_{RUN_LABEL}.csv")
    safe_to_csv(article_meta, output_dir / f"metadata_article_level_{RUN_LABEL}.csv")

    match_candidates, best_matches = build_match_reports(records, article_meta)
    best_match_lookup = best_matches.set_index("text_file").to_dict("index") if not best_matches.empty else {}

    text_output_rows = []
    header_report_rows = []
    for record in records:
        match_row = best_match_lookup.get(record.text_file, {})
        matched_title = str(match_row.get("matched_title", "") or "")
        cleaned, header_stats = clean_readable_text_with_report(record.raw_text, matched_title)
        analysis_text, token_count, analysis_char_count, profile = make_analysis_text(cleaned, record.detected_language, custom_stopwords)
        cleaned_path = cleaned_dir / record.text_file
        analysis_path = analysis_dir / record.text_file
        write_text(cleaned_path, cleaned)
        write_text(analysis_path, analysis_text)
        text_output_rows.append({
            "text_id": record.text_id,
            "text_file": record.text_file,
            "raw_text_path": str(record.raw_text_path),
            "cleaned_readable_text_path": str(cleaned_path),
            "analysis_text_path": str(analysis_path),
            "cleaned_readable_char_count": len(cleaned),
            "analysis_token_count": token_count,
            "analysis_char_count": analysis_char_count,
            "analysis_preprocessing_profile": profile,
            "cleaned_readable_excerpt": excerpt(cleaned[:1500], 500),
            "duplicate_hash": sha1_text(normalize_for_duplicate(cleaned)),
            "text_units_path": str(text_units_dir / "article_text_units.csv"),
        })
        header_report_rows.append({
            "filename": record.text_file,
            "detected_master_id": match_row.get("matched_master_id", ""),
            "matched_title": matched_title,
            "generated_page_marker_lines_removed": header_stats.generated_page_markers_removed,
            "running_header_lines_removed": header_stats.running_header_lines_removed,
            "standalone_page_number_lines_removed": header_stats.standalone_page_numbers_removed,
            "journal_front_matter_removed": header_stats.journal_front_matter_removed,
            "first_10_removed_lines": " | ".join(header_stats.removed_lines or []),
            "warnings_or_uncertainty_flags": "; ".join(sorted(header_stats.warnings or [])),
        })
    text_outputs = pd.DataFrame(text_output_rows)
    header_report = pd.DataFrame(header_report_rows)

    units_diag = build_text_units(text_outputs, records_by_file, text_units_dir / "article_text_units.csv")

    excerpt_lookup = text_outputs.set_index("text_file")["cleaned_readable_excerpt"].to_dict()
    for df in (match_candidates, best_matches):
        if not df.empty:
            df["cleaned_readable_excerpt"] = df["text_file"].map(excerpt_lookup).fillna("")
    safe_to_csv(match_candidates, report_dir / f"text_match_candidates_top5_{RUN_LABEL}.csv")
    safe_to_csv(best_matches, report_dir / f"text_metadata_matches_{RUN_LABEL}.csv")
    safe_to_csv(header_report, report_dir / "header_cleaning_report.csv")

    master = manifest.merge(text_outputs, on=["text_id", "text_file", "raw_text_path"], how="left")
    master = master.merge(units_diag, on="text_file", how="left")
    master = master.merge(best_matches, on=["text_id", "text_file", "detected_language", "final_text_type", "exclude_from_article_analysis"], how="left")
    master = master.merge(
        article_meta.drop(columns=["title_match_norm", "title_match_tokens"]),
        left_on="matched_master_id",
        right_on="master_id",
        how="left",
    )

    duplicate_df = duplicate_groups(master)
    safe_to_csv(duplicate_df, report_dir / f"duplicate_text_groups_{RUN_LABEL}.csv")
    master = add_final_dataset_flags(master, duplicate_df)

    required_order = [
        "text_file", "raw_text_path", "cleaned_readable_text_path", "analysis_text_path",
        "text_units_path", "raw_text_checksum", "raw_char_count",
        "cleaned_readable_char_count", "analysis_token_count", "paragraph_count",
        "sentence_count", "detected_language", "language_confidence",
        "possible_mixed_language_flag", "suggested_text_type", "final_text_type",
        "exclude_from_article_analysis", "exclusion_reason", "matched_master_id",
        "matched_title", "matched_year", "match_confidence", "year_conflict_flag",
        "duplicate_group_id", "duplicate_flag", "needs_match_review_flag",
    ]
    ordered_cols = [c for c in required_order if c in master.columns] + [c for c in master.columns if c not in required_order]
    master = master[ordered_cols]
    master_written_path = safe_to_csv(master, output_dir / f"article_text_master_dataset_{RUN_LABEL}.csv")

    non_article_report_cols = [
        "text_file", "suggested_text_type", "final_text_type", "exclude_from_article_analysis",
        "exclusion_reason", "raw_text_path", "cleaned_readable_text_path",
        "detected_language", "language_confidence", "filename_year_hint",
    ]
    safe_to_csv(
        master[master["exclude_from_article_analysis"].map(truthy)][non_article_report_cols],
        report_dir / f"non_article_exclusion_report_{RUN_LABEL}.csv",
    )
    create_language_distribution(master, report_dir / f"language_distribution_{RUN_LABEL}.csv")

    matched_master_ids = set(master.loc[master["match_confidence"].isin(["high", "medium"]), "matched_master_id"].dropna().astype(str))
    unmatched_metadata = article_meta[~article_meta["master_id"].astype(str).isin(matched_master_ids)].drop(columns=["title_match_norm", "title_match_tokens"])
    safe_to_csv(unmatched_metadata, report_dir / f"unmatched_metadata_articles_{RUN_LABEL}.csv")

    review_texts = master[master["needs_match_review_flag"]].copy()
    safe_to_csv(review_texts, report_dir / f"texts_needing_match_review_{RUN_LABEL}.csv")

    final_dataset = master[master["include_in_final_article_analysis"]].copy()
    final_written_path = safe_to_csv(final_dataset, output_dir / f"final_article_analysis_dataset_{RUN_LABEL}.csv")
    exclusion_report = master[~master["include_in_final_article_analysis"]].copy()
    safe_to_csv(exclusion_report, report_dir / f"final_dataset_exclusion_report_{RUN_LABEL}.csv")

    figures = make_plots(master, figures_dir)
    write_corpus_diagnostics(master, duplicate_df, figures, report_dir / f"corpus_diagnostics_{RUN_LABEL}.md")
    final_ready, tfidf = prepare_analysis_ready(master, analysis_ready_dir)
    write_preliminary_analysis_summary(final_ready, tfidf, report_dir / f"preliminary_analysis_summary_{RUN_LABEL}.md")
    write_text_version_diagnostics(master, report_dir / f"text_version_diagnostics_{RUN_LABEL}.md", records_by_file)
    write_header_cleaning_samples(records, master, header_report, report_dir / "header_cleaning_samples.md")

    summary = {
        "metadata_rows": int(len(metadata)),
        "metadata_unique_articles": int(article_meta["master_id"].nunique()),
        "text_files": int(len(records)),
        "manifest_rows": int(len(manifest)),
        "matches_by_confidence": {str(k): int(v) for k, v in Counter(master["match_confidence"].fillna("missing")).items()},
        "exact_duplicate_groups": int(duplicate_df["duplicate_group_id"].nunique()) if not duplicate_df.empty else 0,
        "excluded_non_articles_or_uncertain": int(master["exclude_from_article_analysis"].map(truthy).sum()),
        "final_article_analysis_texts": int(len(final_dataset)),
        "languages": {str(k): int(v) for k, v in Counter(master["detected_language"].fillna("missing")).items()},
        "output_dir": str(output_dir),
        "article_text_master_dataset_path": str(master_written_path),
        "final_article_analysis_dataset_path": str(final_written_path),
        "header_cleaning": {
            "generated_page_marker_lines_removed": int(header_report["generated_page_marker_lines_removed"].sum()),
            "running_header_lines_removed": int(header_report["running_header_lines_removed"].sum()),
            "standalone_page_number_lines_removed": int(header_report["standalone_page_number_lines_removed"].sum()),
            "files_with_front_matter_removed": int(header_report["journal_front_matter_removed"].map(truthy).sum()),
            "files_with_uncertainty_warnings": int(header_report["warnings_or_uncertainty_flags"].astype(str).str.len().gt(0).sum()),
        },
    }
    (report_dir / f"pipeline_summary_{RUN_LABEL}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_transformation_log(project_dir, output_dir, summary, run_label=RUN_LABEL)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RAS article text dataset.")
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=DEFAULT_PROJECT_DIR,
        help="Folder containing articles/, ras_subject_expanded.csv, and hit_stopwords.txt.",
    )
    parser.add_argument(
        "--run-header-cleaning-tests",
        action="store_true",
        help="Run the built-in header-cleaning unit tests and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_header_cleaning_tests:
        run_header_cleaning_tests()
        return
    summary = build_dataset(args.project_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
