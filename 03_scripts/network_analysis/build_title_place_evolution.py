from __future__ import annotations

import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(r"C:\ras_text_analysis")
EXPANDED_CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "namecorrected_lcsh_outputs" / "ras_subject_expanded.csv"
OUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "namecorrected_lcsh_outputs"
SUFFIX = sys.argv[3] if len(sys.argv) > 3 else "_namecorrected"

PLACE_TIME_CSV = OUT_DIR / f"title_place_time_subject_expanded{SUFFIX}.csv"
PLACE_SUBJECT_TIME_CSV = OUT_DIR / f"title_place_subject_time_subject_expanded{SUFFIX}.csv"
ARTICLE_PLACES_CSV = OUT_DIR / f"article_title_places_subject_expanded{SUFFIX}.csv"
PLACE_MAPPING_CSV = OUT_DIR / f"title_place_name_mapping_subject_expanded{SUFFIX}.csv"


MANUAL_PLACES = {
    "batang": "Batang",
    "cambodia": "Cambodia",
    "canton": "Guangzhou",
    "cheh-kiang": "Zhejiang Province",
    "ch'eng tu": "Chengdu",
    "cheng tu": "Chengdu",
    "chinkiang": "Zhenjiang",
    "foochow": "Fuzhou",
    "formosa": "Taiwan",
    "fukien": "Fujian Province",
    "hangchow": "Hangzhou",
    "hangkow": "Hangzhou",
    "hankow": "Hankou",
    "honan": "Henan Province",
    "huangho": "Yellow River",
    "hunan": "Hunan",
    "kaifungfu": "Kaifengfu",
    "kansu": "Gansu Province",
    "kambodia": "Cambodia",
    "kwang tung": "Guangdong Province",
    "kwang-tung": "Guangdong Province",
    "kwangtung": "Guangdong Province",
    "kwangsi": "Guangxi Province",
    "kweichow": "Guizhou",
    "lampacao": "Lampacau",
    "menkong": "Mangkang",
    "pekin": "Beijing",
    "quang-tung": "Guangdong Province",
    "quang tang": "Guangdong Province",
    "rangoon": "Yangon",
    "saghalien": "Sakhalin",
    "shan tung": "Shandong Province",
    "shan-hsi": "Shanxi Province",
    "shantung": "Shandong Province",
    "shen-hsi": "Shaanxi Province",
    "shaohing": "Shaoxing",
    "ssuch'uan": "Sichuan",
    "sungp'an": "Songpan",
    "sze chuen": "Sichuan Province",
    "szechuen": "Sichuan",
    "szemao": "Pu'er",
    "t'ai shan": "Tai Shan",
    "tai shan": "Tai Shan",
    "tong-king": "Tonkin",
    "wenchow": "Wenzhou",
    "woosung": "Wusong",
    "yang - tsze kiang": "Yangtzi River",
    "yang-tsze kiang": "Yangtzi River",
    "yangtse": "Yangtzi River",
    "yentai hill": "Yantai Hill",
}

CANONICAL_MODERN_NAMES = {
    "chengdu plain": "Chengdu",
    "guangdong province": "Guangdong",
    "shandong province": "Shandong",
    "sichuan province": "Sichuan",
}

PLACE_COORDS = {
    "Batang": (30.0, 99.1),
    "Beijing": (39.9, 116.4),
    "Cambodia": (12.6, 104.9),
    "Chengdu": (30.7, 104.1),
    "Fujian Province": (26.1, 117.9),
    "Fuzhou": (26.1, 119.3),
    "Gansu Province": (36.1, 103.8),
    "Guangdong": (23.1, 113.3),
    "Guangxi Province": (22.8, 108.3),
    "Guangzhou": (23.1, 113.3),
    "Guizhou": (26.6, 106.7),
    "Hankou": (30.6, 114.3),
    "Henan Province": (34.8, 113.6),
    "Hunan": (28.2, 112.9),
    "Kaifengfu": (34.8, 114.3),
    "Lampacau": (22.2, 113.5),
    "Mangkang": (29.7, 98.6),
    "Pu'er": (22.8, 100.9),
    "Sakhalin": (50.7, 143.0),
    "Shaanxi Province": (34.3, 108.9),
    "Shandong": (36.7, 117.0),
    "Shanxi Province": (37.9, 112.5),
    "Shaoxing": (30.0, 120.6),
    "Sichuan": (30.7, 104.1),
    "Songpan": (32.7, 103.6),
    "Tai Shan": (36.3, 117.1),
    "Taiwan": (23.7, 121.0),
    "Tonkin": (21.0, 105.8),
    "Wenzhou": (28.0, 120.7),
    "Wusong": (31.4, 121.5),
    "Yangon": (16.8, 96.2),
    "Yangtzi River": (30.6, 114.3),
    "Yantai Hill": (37.5, 121.4),
    "Zhenjiang": (32.2, 119.5),
}

PLACE_HINTS = (
    "province",
    "river",
    "island",
    "plain",
    "city",
    "hill",
    "valley",
    "coast",
    "wall",
    "fu",
)

NON_PLACE_HINTS = (
    "dynasty",
    "book",
    "dream of",
    "church",
    "gong",
    "tang",
    "tribe",
    "zu",
    "zi",
    "emperor",
    "commander",
)


def clean_note_text(text: str) -> str:
    text = str(text or "").replace("<U+FF1B>", ";").replace("<U+FF0C>", ",")
    text = re.sub(r"<U\+[0-9A-Fa-f]+>", " ", str(text or ""))
    text = text.replace("；", ";").replace("，", ",")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_match(text: str) -> str:
    text = str(text or "").lower()
    text = text.replace("ü", "u")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_name(name: str) -> str:
    name = re.sub(r"<U\+[0-9A-Fa-f]+>", " ", str(name or ""))
    name = re.sub(r"\b(currently|currently known as|the old name of|old name of)\b", " ", name, flags=re.I)
    name = re.sub(r"\bnorthern part of\b.*$", " ", name, flags=re.I)
    name = re.sub(r"\s+", " ", name)
    return name.strip(" .,*")


def canonical_modern_name(name: str) -> str:
    cleaned = clean_name(name)
    return CANONICAL_MODERN_NAMES.get(normalize_match(cleaned), cleaned)


def place_label(modern_name: str, historical_names: list[str]) -> str:
    unique_old = []
    modern_norm = normalize_match(modern_name)
    for old in historical_names:
        old = clean_name(old)
        if not old or normalize_match(old) == modern_norm:
            continue
        if old not in unique_old:
            unique_old.append(old)
    if unique_old:
        return f"{modern_name}（{'; '.join(unique_old)}）"
    return modern_name


def place_coords(modern_name: str) -> tuple[float | None, float | None]:
    return PLACE_COORDS.get(modern_name, (None, None))


def title_contains(title_norm: str, candidate: str) -> bool:
    cand = normalize_match(candidate)
    if not cand:
        return False
    if cand in title_norm:
        return True
    compact_title = title_norm.replace(" ", "")
    compact_cand = cand.replace(" ", "")
    return len(compact_cand) >= 5 and compact_cand in compact_title


def infer_new_name(old_name: str, parts: list[str]) -> str:
    manual = MANUAL_PLACES.get(normalize_match(old_name))
    if manual:
        return manual
    for raw in parts[1:]:
        candidate = clean_name(raw)
        if not candidate:
            continue
        m = re.search(r"old name of\s+([A-Za-z][A-Za-z '\-]+)", raw, flags=re.I)
        if m:
            return clean_name(m.group(1))
        m = re.search(r"currently\s+([A-Za-z][A-Za-z '\-]+)", raw, flags=re.I)
        if m:
            return clean_name(m.group(1))
        if re.search(r"[A-Za-z]", candidate):
            return candidate
    return clean_name(old_name)


def looks_like_place(old_name: str, new_name: str) -> bool:
    old_norm = normalize_match(old_name)
    new_norm = normalize_match(new_name)
    if old_norm in MANUAL_PLACES:
        return True
    combined = f"{old_norm} {new_norm}"
    if any(h in combined for h in PLACE_HINTS):
        return True
    if any(h in combined for h in NON_PLACE_HINTS):
        return False
    return old_norm in {
        "canton",
        "formosa",
        "hankow",
        "hunan",
        "pekin",
        "shantung",
        "kwangsi",
        "kweichow",
        "foochow",
        "rangoon",
        "honan",
        "kansu",
    }


def extract_note_place_candidates(notes: str) -> list[dict[str, str]]:
    cleaned = clean_note_text(notes)
    if not cleaned:
        return []
    candidates = []
    for segment in re.split(r";", cleaned):
        segment = segment.strip()
        if not segment:
            continue
        parts = [clean_name(p) for p in segment.split(",") if clean_name(p)]
        if not parts:
            continue

        # Some notes omit the comma between old and new: "YANG-TSZE KIANG Yangtzi River".
        first = parts[0]
        manual_old = None
        for old in sorted(MANUAL_PLACES, key=len, reverse=True):
            if normalize_match(old) and normalize_match(old) in normalize_match(first):
                manual_old = old
                break
        old_name = manual_old or first
        new_name = infer_new_name(old_name, parts)
        if looks_like_place(old_name, new_name):
            old_clean = clean_name(old_name)
            new_clean = canonical_modern_name(new_name)
            candidates.append(
                {
                    "historical_name": old_clean,
                    "modern_name": new_clean,
                    "place_label": place_label(new_clean, [old_clean]),
                }
            )
    return candidates


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(EXPANDED_CSV, encoding="utf-8")
    df["year_start"] = pd.to_numeric(df["year_start"], errors="coerce")
    df = df[
        (df["subject_l1"] != "Unknown")
        & (df["subject_l2"] != "Unknown")
        & df["subject_l1"].notna()
        & df["subject_l2"].notna()
    ].copy()
    df = df.drop_duplicates(subset=["master_id", "subject_l2"])
    articles = df[
        ["master_id", "time_window", "author_raw", "title", "year_start", "notes"]
    ].drop_duplicates("master_id")
    subjects = df[["master_id", "time_window", "subject_l1", "subject_l2"]].drop_duplicates()
    return articles, subjects


def add_shares(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    totals = df.groupby(group_cols)["place_count"].sum().rename("window_place_total")
    df = df.merge(totals, on=group_cols, how="left")
    df["share"] = df["place_count"] / df["window_place_total"]
    return df.drop(columns=["window_place_total"])


def add_tfidf(df: pd.DataFrame, place_col: str = "place_label") -> pd.DataFrame:
    windows = sorted(df["time_window"].dropna().unique())
    n_windows = len(windows)
    doc_freq = df.groupby(place_col)["time_window"].nunique().to_dict()
    df = df.copy()
    df["idf"] = df[place_col].map(lambda p: math.log((1 + n_windows) / (1 + doc_freq.get(p, 0))) + 1)
    df["tfidf"] = df["place_count"] * df["idf"]
    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    articles, article_subjects = load_data()

    article_place_rows = []
    mapping_rows = []
    for row in articles.to_dict("records"):
        title_norm = normalize_match(row["title"])
        notes_norm = normalize_match(row.get("notes", ""))
        seen_labels = set()
        candidates = extract_note_place_candidates(row.get("notes", ""))
        for old_name, modern_name in MANUAL_PLACES.items():
            if title_contains(title_norm, old_name) and title_contains(notes_norm, old_name):
                modern_name = canonical_modern_name(modern_name)
                candidates.append(
                    {
                        "historical_name": old_name,
                        "modern_name": modern_name,
                        "place_label": place_label(modern_name, [old_name]),
                    }
                )
        by_modern: dict[str, set[str]] = defaultdict(set)
        for cand in candidates:
            if title_contains(title_norm, cand["historical_name"]) or title_contains(title_norm, cand["modern_name"]):
                modern = canonical_modern_name(cand["modern_name"])
                by_modern[modern].add(clean_name(cand["historical_name"]))

        for modern, historical_set in by_modern.items():
            if modern in seen_labels:
                continue
            seen_labels.add(modern)
            historical_names = sorted(historical_set, key=lambda x: normalize_match(x))
            label = place_label(modern, historical_names)
            lat, lon = place_coords(modern)
            record = {
                "historical_name": "; ".join(historical_names),
                "modern_name": modern,
                "place_label": label,
                "lat": lat,
                "lon": lon,
            }
            article_place_rows.append(
                {
                    "master_id": row["master_id"],
                    "time_window": row["time_window"],
                    "author_raw": row["author_raw"],
                    "title": row["title"],
                    "year_start": row["year_start"],
                    **record,
                }
            )
            mapping_rows.append(record)

    article_places = pd.DataFrame(article_place_rows)
    if article_places.empty:
        article_places = pd.DataFrame(
            columns=[
                "master_id",
                "time_window",
                "author_raw",
                "title",
                "year_start",
                "historical_name",
                "modern_name",
                "place_label",
                "lat",
                "lon",
            ]
        )
    place_mapping = pd.DataFrame(mapping_rows).drop_duplicates().sort_values(["modern_name", "historical_name"])

    place_time_rows = []
    for (window, modern), group in article_places.groupby(["time_window", "modern_name"]):
        historical = sorted(
            {h.strip() for value in group["historical_name"].dropna() for h in str(value).split(";") if h.strip()},
            key=lambda x: normalize_match(x),
        )
        lat, lon = place_coords(modern)
        place_time_rows.append(
            {
                "time_window": window,
                "place_label": place_label(modern, historical),
                "modern_name": modern,
                "historical_name": "; ".join(historical),
                "lat": lat,
                "lon": lon,
                "place_count": len(group),
                "article_count": group["master_id"].nunique(),
            }
        )
    place_time = pd.DataFrame(place_time_rows).sort_values(["time_window", "place_count"], ascending=[True, False])
    if not place_time.empty:
        place_time = add_tfidf(add_shares(place_time, ["time_window"]), "place_label")

    place_subject = article_places.merge(article_subjects, on=["master_id", "time_window"], how="inner")
    place_subject_rows = []
    for (window, subject_l1, subject_l2, modern), group in place_subject.groupby(["time_window", "subject_l1", "subject_l2", "modern_name"]):
        historical = sorted(
            {h.strip() for value in group["historical_name"].dropna() for h in str(value).split(";") if h.strip()},
            key=lambda x: normalize_match(x),
        )
        lat, lon = place_coords(modern)
        place_subject_rows.append(
            {
                "time_window": window,
                "subject_l1": subject_l1,
                "subject_l2": subject_l2,
                "place_label": place_label(modern, historical),
                "modern_name": modern,
                "historical_name": "; ".join(historical),
                "lat": lat,
                "lon": lon,
                "place_count": len(group),
                "article_count": group["master_id"].nunique(),
            }
        )
    place_subject_time = pd.DataFrame(place_subject_rows).sort_values(["time_window", "subject_l1", "place_count"], ascending=[True, True, False])
    if not place_subject_time.empty:
        place_subject_time = add_tfidf(add_shares(place_subject_time, ["time_window", "subject_l1", "subject_l2"]), "place_label")

    article_places.to_csv(ARTICLE_PLACES_CSV, index=False, encoding="utf-8")
    place_time.to_csv(PLACE_TIME_CSV, index=False, encoding="utf-8")
    place_subject_time.to_csv(PLACE_SUBJECT_TIME_CSV, index=False, encoding="utf-8")
    place_mapping.to_csv(PLACE_MAPPING_CSV, index=False, encoding="utf-8")

    print(f"articles_with_places={article_places['master_id'].nunique() if not article_places.empty else 0}")
    print(f"article_place_rows={len(article_places)} place_time={len(place_time)} place_subject_time={len(place_subject_time)}")
    print(f"wrote {PLACE_TIME_CSV}")


if __name__ == "__main__":
    main()
