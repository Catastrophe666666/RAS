import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from difflib import SequenceMatcher

import pandas as pd


ROOT = Path(r"C:\ras_text_analysis")
MERGED_PATH = ROOT / "RAS_merged_fixed1890-1948_xlsx.xlsx"
TOC_PATH = ROOT / "RAS_TOC_all_volumes(1) 的副本.xlsx"
OUT_DIR = ROOT / "outputs" / "ras_reconciliation"
OUT_JSON = OUT_DIR / "ras_reconciliation_data.json"


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "into", "is", "its", "of", "on", "or", "the", "to", "with", "without",
    "part", "parts", "no", "nos", "vol", "volume", "notes", "note",
}


def clean_value(value):
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def excel_value(value):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    return value


def expand_years(value):
    text = clean_value(value)
    if not text:
        return set()

    full = [int(x) for x in re.findall(r"\d{4}", text)]
    if len(full) >= 2:
        return set(range(full[0], full[1] + 1))

    short = re.search(r"(\d{4})\s*[-/]\s*(\d{2})", text)
    if short:
        start = int(short.group(1))
        end = (start // 100) * 100 + int(short.group(2))
        if end < start:
            end += 100
        return set(range(start, end + 1))

    if len(full) == 1:
        return {full[0]}

    return set()


def merged_years(row):
    ys = row.get("year_start")
    ye = row.get("year_end")
    try:
        if pd.notna(ys) and pd.notna(ye):
            return set(range(int(ys), int(ye) + 1))
    except Exception:
        pass
    return expand_years(row.get("year"))


def title_before_author(text):
    text = clean_value(text)
    text = re.sub(r"^\s*ARTICLE\s+[IVXLCDM]+\s*\.?\s*", "", text, flags=re.I)
    text = re.sub(r"\s+\b[Bb]y\b\s+.*$", "", text)
    text = re.sub(r"\.{2,}.*$", "", text)
    return text.strip()


def normalize_title(text):
    text = title_before_author(text).lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.replace("&", " and ")
    text = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", text)
    text = re.sub(r"\b(article|appendix)\s+[ivxlcdm]+\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 1]
    return tokens


def fuzzy_overlap(left_tokens, right_tokens):
    remaining = list(right_tokens)
    common = 0
    for token in left_tokens:
        best_index = None
        best_score = 0
        for idx, candidate in enumerate(remaining):
            score = SequenceMatcher(None, token, candidate).ratio()
            if score > best_score:
                best_score = score
                best_index = idx
        if best_score >= 0.86:
            common += 1
            remaining.pop(best_index)
    return common


def score_titles(left, right):
    lt = normalize_title(left)
    rt = normalize_title(right)
    if not lt or not rt:
        return 0.0

    lc = Counter(lt)
    rc = Counter(rt)
    exact_common = sum((lc & rc).values())
    fuzzy_common = fuzzy_overlap(lt, rt)
    common = max(exact_common, fuzzy_common)
    precision = common / max(1, sum(rc.values()))
    recall = common / max(1, sum(lc.values()))
    f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    l_sorted = " ".join(sorted(lt))
    r_sorted = " ".join(sorted(rt))
    seq = SequenceMatcher(None, l_sorted, r_sorted).ratio()

    contain = 0.0
    lset = set(lt)
    rset = set(rt)
    if lset and (lset <= rset or rset <= lset):
        contain = 1.0
    min_len = min(len(lt), len(rt))
    if min_len >= 4 and common >= 3 and common / min_len >= 0.75:
        contain = max(contain, 0.98)

    return round(max(0.72 * f1 + 0.28 * seq, contain * 0.94), 4)


def status_for(score, title):
    token_count = len(set(normalize_title(title)))
    if score >= 0.92 and token_count >= 2:
        return "accepted"
    if score >= 0.82 and token_count >= 3:
        return "review"
    return "needs_manual_check"


def best_match(row, candidates, title_field):
    best = None
    best_score = -1
    for _, cand in candidates.iterrows():
        score = score_titles(row[title_field], cand["title"])
        if score > best_score:
            best_score = score
            best = cand
    return best, best_score


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    merged = pd.read_excel(MERGED_PATH)
    toc = pd.read_excel(TOC_PATH, sheet_name="All_TOCs")

    merged["_row_id"] = [i + 2 for i in range(len(merged))]
    toc["_row_id"] = [i + 2 for i in range(len(toc))]
    merged["_years"] = merged.apply(merged_years, axis=1)
    toc["_years"] = toc["year"].map(expand_years)

    merged_rows = []
    for _, row in merged.iterrows():
        years = row["_years"]
        candidates = toc[toc["_years"].map(lambda ys: bool(years & ys))]
        match, score = best_match(row, candidates, "title") if not candidates.empty else (None, 0)
        status = "no_toc_year_candidate"
        toc_page = toc_title = toc_pdf = toc_year = ""
        if match is not None:
            status = status_for(score, row["title"])
            toc_page = match.get("page_number")
            toc_title = match.get("title")
            toc_pdf = match.get("source_pdf")
            toc_year = match.get("year")

        out = {c: excel_value(row[c]) for c in merged.columns if not c.startswith("_")}
        out["page_original"] = excel_value(row.get("page"))
        if status == "accepted" and clean_value(toc_page):
            out["page"] = excel_value(toc_page)
        out["toc_page_number"] = excel_value(toc_page)
        out["toc_title"] = excel_value(toc_title)
        out["toc_source_pdf"] = excel_value(toc_pdf)
        out["toc_year"] = excel_value(toc_year)
        out["title_match_score"] = score
        out["match_status"] = status
        out["merged_source_row"] = int(row["_row_id"])
        merged_rows.append(out)

    toc_rows = []
    for _, row in toc.iterrows():
        years = row["_years"]
        candidates = merged[merged["_years"].map(lambda ys: bool(years & ys))]
        match, score = best_match(row, candidates, "title") if not candidates.empty else (None, 0)
        status = "no_merged_year_candidate"
        author = subject = merged_title = merged_page = ""
        if match is not None:
            status = status_for(score, row["title"])
            author = match.get("author")
            subject = match.get("subject")
            merged_title = match.get("title")
            merged_page = match.get("page")

        out = {c: excel_value(row[c]) for c in toc.columns if not c.startswith("_")}
        out["author_from_merged"] = excel_value(author) if status == "accepted" else None
        out["subject_from_merged"] = excel_value(subject) if status == "accepted" else None
        out["merged_title"] = excel_value(merged_title)
        out["merged_page_original"] = excel_value(merged_page)
        out["title_match_score"] = score
        out["match_status"] = status
        out["toc_source_row"] = int(row["_row_id"])
        toc_rows.append(out)

    review_rows = []
    for row in merged_rows:
        if row["match_status"] != "accepted":
            review_rows.append({
                "direction": "RAS_merged needs TOC page",
                "source_row": row["merged_source_row"],
                "year": row.get("year"),
                "volume": row.get("volume"),
                "merged_title": row.get("title"),
                "toc_title": row.get("toc_title"),
                "old_page": row.get("page_original"),
                "toc_page": row.get("toc_page_number"),
                "author": row.get("author"),
                "subject": row.get("subject"),
                "score": row.get("title_match_score"),
                "status": row.get("match_status"),
            })
    for row in toc_rows:
        if row["match_status"] != "accepted":
            review_rows.append({
                "direction": "RAS_TOC needs merged author/subject",
                "source_row": row["toc_source_row"],
                "year": row.get("year"),
                "volume": row.get("source_pdf"),
                "merged_title": row.get("merged_title"),
                "toc_title": row.get("title"),
                "old_page": row.get("merged_page_original"),
                "toc_page": row.get("page_number"),
                "author": row.get("author_from_merged"),
                "subject": row.get("subject_from_merged"),
                "score": row.get("title_match_score"),
                "status": row.get("match_status"),
            })

    summary = [
        ["RAS_merged rows", len(merged_rows)],
        ["RAS_merged accepted page replacements", sum(r["match_status"] == "accepted" for r in merged_rows)],
        ["RAS_merged review/manual rows", sum(r["match_status"] != "accepted" for r in merged_rows)],
        ["RAS_TOC rows", len(toc_rows)],
        ["RAS_TOC accepted author/subject fills", sum(r["match_status"] == "accepted" for r in toc_rows)],
        ["RAS_TOC review/manual rows", sum(r["match_status"] != "accepted" for r in toc_rows)],
    ]

    data = {
        "summary": summary,
        "merged_headers": list(merged_rows[0].keys()) if merged_rows else [],
        "merged_rows": merged_rows,
        "toc_headers": list(toc_rows[0].keys()) if toc_rows else [],
        "toc_rows": toc_rows,
        "review_headers": list(review_rows[0].keys()) if review_rows else [],
        "review_rows": review_rows,
    }
    OUT_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT_JSON)
    print(json.dumps(dict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
