"""
Extract the scanned RAS index PDF into CSV using Google Vision OCR.

Output columns:
    year, volume, author, title, page, subject, status, remark

The index is assumed to span two facing pages per entry: odd-numbered PDF
pages hold the left-side columns, even-numbered PDF pages hold the right-side
columns. OCR JSON is cached so repeated parsing does not call the API again.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import fitz
import pandas as pd


ROOT = Path(r"C:\ras_text_analysis")
PDF_PATH = ROOT / "index001.pdf"
KEY_PATH = ROOT / "key1.txt"
CACHE_DIR = ROOT / "ocr_cache" / "index001_google_vision"
OUT_CSV = ROOT / "index001_extracted.csv"
REVIEW_CSV = ROOT / "index001_extraction_review.csv"
OCR_PDF = ROOT / "index001_ocr.pdf"

OUTPUT_COLUMNS = ["year", "volume", "author", "title", "page", "subject", "status", "remark"]


@dataclass
class Word:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def xc(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def yc(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass
class Block:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def xc(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def yc(self) -> float:
        return (self.y0 + self.y1) / 2


def read_api_key(path: Path) -> str:
    key = path.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError(f"API key file is empty: {path}")
    return key


def render_page_png(page: fitz.Page, zoom: float = 2.5) -> bytes:
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return pix.tobytes("png")


def call_google_vision(image_bytes: bytes, api_key: str, retries: int = 3) -> dict:
    url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
    payload = {
        "requests": [
            {
                "image": {"content": base64.b64encode(image_bytes).decode("ascii")},
                "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
                "imageContext": {"languageHints": ["en"]},
            }
        ]
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                result = json.loads(response.read().decode("utf-8"))
            page_result = result["responses"][0]
            if "error" in page_result:
                raise RuntimeError(page_result["error"])
            return page_result
        except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            if attempt == retries:
                raise RuntimeError(f"Google Vision request failed after {retries} attempts: {exc}") from exc
            time.sleep(2 * attempt)
    raise RuntimeError("Google Vision request failed")


def ensure_ocr_cache(pdf_path: Path, key_path: Path, cache_dir: Path, limit_pages: int | None = None) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    api_key = read_api_key(key_path)
    doc = fitz.open(pdf_path)
    total = len(doc) if limit_pages is None else min(limit_pages, len(doc))

    for page_index in range(total):
        cache_path = cache_dir / f"page_{page_index + 1:03d}.json"
        if cache_path.exists():
            print(f"OCR cache exists: page {page_index + 1}/{total}")
            continue

        print(f"OCR Google Vision: page {page_index + 1}/{total}")
        image_bytes = render_page_png(doc[page_index])
        page_result = call_google_vision(image_bytes, api_key)
        cache_path.write_text(json.dumps(page_result, ensure_ascii=False), encoding="utf-8")


def vertices_to_box(vertices: list[dict]) -> tuple[float, float, float, float]:
    xs = [v.get("x", 0) for v in vertices]
    ys = [v.get("y", 0) for v in vertices]
    return min(xs), min(ys), max(xs), max(ys)


def words_from_vision_json(data: dict) -> list[Word]:
    words: list[Word] = []
    for page in data.get("fullTextAnnotation", {}).get("pages", []):
        for block in page.get("blocks", []):
            for paragraph in block.get("paragraphs", []):
                for word in paragraph.get("words", []):
                    text = "".join(symbol.get("text", "") for symbol in word.get("symbols", []))
                    if not text.strip():
                        continue
                    x0, y0, x1, y1 = vertices_to_box(word.get("boundingBox", {}).get("vertices", []))
                    words.append(Word(text=text, x0=x0, y0=y0, x1=x1, y1=y1))
    return words


def blocks_from_vision_json(data: dict) -> list[Block]:
    blocks: list[Block] = []
    for page in data.get("fullTextAnnotation", {}).get("pages", []):
        for block in page.get("blocks", []):
            parts = []
            for paragraph in block.get("paragraphs", []):
                words = []
                for word in paragraph.get("words", []):
                    words.append("".join(symbol.get("text", "") for symbol in word.get("symbols", [])))
                if words:
                    parts.append(" ".join(words))
            text = clean_cell(" ".join(parts))
            if not text:
                continue
            x0, y0, x1, y1 = vertices_to_box(block.get("boundingBox", {}).get("vertices", []))
            blocks.append(Block(text=text, x0=x0, y0=y0, x1=x1, y1=y1))
    return blocks


def load_page_words(cache_dir: Path, page_num: int) -> list[Word]:
    path = cache_dir / f"page_{page_num:03d}.json"
    return words_from_vision_json(json.loads(path.read_text(encoding="utf-8")))


def load_page_blocks(cache_dir: Path, page_num: int) -> list[Block]:
    path = cache_dir / f"page_{page_num:03d}.json"
    return blocks_from_vision_json(json.loads(path.read_text(encoding="utf-8")))


def load_page_text(cache_dir: Path, page_num: int) -> str:
    path = cache_dir / f"page_{page_num:03d}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("fullTextAnnotation", {}).get("text", "")


def page_size(words: list[Word]) -> tuple[float, float]:
    if not words:
        return 1.0, 1.0
    return max(w.x1 for w in words), max(w.y1 for w in words)


def is_noise_line(text: str) -> bool:
    compact = re.sub(r"\s+", " ", text.strip()).lower()
    if not compact:
        return True
    if compact in {"year volume author title", "year vol author title", "page subject status remark"}:
        return True
    if compact.startswith("index") and len(compact) < 30:
        return True
    return False


def find_header_y(words: list[Word], expected: Iterable[str]) -> float | None:
    expected_l = set(expected)
    candidates = [w for w in words if clean_token(w.text).lower().rstrip(".") in expected_l]
    if not candidates:
        return None
    ys = sorted(w.yc for w in candidates)
    return ys[0]


def clean_token(text: str) -> str:
    return re.sub(r"^[^\w]+|[^\w.]+$", "", text)


def column_bounds(words: list[Word], side: str) -> dict[str, tuple[float, float]]:
    width, _height = page_size(words)
    if side == "left":
        names = ["year", "volume", "author", "title"]
        header_alias = {
            "year": {"year"},
            "volume": {"volume", "vol"},
            "author": {"author", "authors"},
            "title": {"title"},
        }
        fallback = {
            "year": (0.02 * width, 0.16 * width),
            "volume": (0.16 * width, 0.28 * width),
            "author": (0.28 * width, 0.50 * width),
            "title": (0.50 * width, 0.98 * width),
        }
    else:
        names = ["page", "subject", "status", "remark"]
        header_alias = {
            "page": {"page", "pages"},
            "subject": {"subject", "subjects"},
            "status": {"status"},
            "remark": {"remark", "remarks", "notes"},
        }
        fallback = {
            "page": (0.02 * width, 0.18 * width),
            "subject": (0.18 * width, 0.55 * width),
            "status": (0.55 * width, 0.75 * width),
            "remark": (0.75 * width, 0.98 * width),
        }

    anchors: dict[str, float] = {}
    for name in names:
        matches = [
            w
            for w in words
            if clean_token(w.text).lower().rstrip(".") in header_alias[name]
            and w.yc < page_size(words)[1] * 0.25
        ]
        if matches:
            anchors[name] = sorted(matches, key=lambda w: w.yc)[0].xc

    if len(anchors) < 3:
        return fallback

    ordered = [(name, anchors.get(name, (fallback[name][0] + fallback[name][1]) / 2)) for name in names]
    centers = [x for _name, x in ordered]
    bounds: dict[str, tuple[float, float]] = {}
    for i, name in enumerate(names):
        left = 0 if i == 0 else (centers[i - 1] + centers[i]) / 2
        right = width if i == len(names) - 1 else (centers[i] + centers[i + 1]) / 2
        bounds[name] = (left, right)
    return bounds


def group_words_by_line(words: list[Word], y_gap: float = 16.0) -> list[list[Word]]:
    if not words:
        return []
    lines: list[list[Word]] = []
    for word in sorted(words, key=lambda w: (w.yc, w.xc)):
        if not lines:
            lines.append([word])
            continue
        prev_y = sum(w.yc for w in lines[-1]) / len(lines[-1])
        if abs(word.yc - prev_y) <= y_gap:
            lines[-1].append(word)
        else:
            lines.append([word])
    return [sorted(line, key=lambda w: w.xc) for line in lines]


def line_to_columns(line: list[Word], bounds: dict[str, tuple[float, float]]) -> dict[str, str]:
    values = {name: [] for name in bounds}
    for word in line:
        for name, (x0, x1) in bounds.items():
            if x0 <= word.xc < x1:
                values[name].append(word.text)
                break
    return {name: " ".join(parts).strip() for name, parts in values.items()}


def clean_cell(value: str) -> str:
    value = re.sub(r"\s+", " ", value or "").strip()
    value = value.replace(" ,", ",").replace(" .", ".").replace("( ", "(").replace(" )", ")")
    return value


def looks_like_year(value: str) -> bool:
    return bool(re.search(r"\b(18|19|20)\d{2}(\s*-\s*\d{2,4})?\b", value or ""))


def looks_like_volume(value: str) -> bool:
    return bool(re.search(r"\b([ivxlcdm]+|\d+)\b", (value or "").lower()))


def looks_like_page(value: str) -> bool:
    return bool(re.search(r"\d", value or ""))


def is_year_line(value: str) -> bool:
    return bool(re.fullmatch(r"(18|19|20)\d{2}(\s*-\s*\d{2,4})?|\d{2}\s+(18|19|20)\d{2}\s*-", clean_cell(value)))


def normalize_year(value: str) -> str:
    value = clean_cell(value)
    value = re.sub(r"\s*-\s*", "-", value)
    flipped = re.fullmatch(r"(\d{2})\s+((18|19|20)\d{2})-", value)
    if flipped:
        return f"{flipped.group(2)}-{flipped.group(1)}"
    return value


def is_volume_line(value: str) -> bool:
    value = clean_cell(value)
    return bool(re.fullmatch(r"(OS|NS)?\s*\d+(\s+Part\s+\d+)?|[IVXLCDM]+", value, flags=re.I))


def is_page_line(value: str) -> bool:
    value = clean_cell(value)
    return bool(re.fullmatch(r"(\d+[a-z]?|\d+\s*[-–]\s*\d+[a-z]?|[ivxlcdm]+\s*[-–]\s*[ivxlcdm]+)", value, flags=re.I))


STATUS_WORDS = {
    "missing",
    "lent",
    "available",
    "present",
    "bound",
    "reprint",
    "photocopy",
    "copy",
    "checked",
    "unknown",
}


def is_status_line(value: str) -> bool:
    value_l = clean_cell(value).lower()
    if value_l in STATUS_WORDS:
        return True
    return bool(re.fullmatch(r"(missing|lent|available|present|unknown)(\s+.*)?", value_l))


def clean_ocr_lines(text: str) -> list[str]:
    lines = [clean_cell(line) for line in text.splitlines()]
    lines = [line for line in lines if line]
    return lines


def block_page_size(blocks: list[Block]) -> tuple[float, float]:
    if not blocks:
        return 1.0, 1.0
    return max(block.x1 for block in blocks), max(block.y1 for block in blocks)


def block_column_bounds(blocks: list[Block], side: str) -> dict[str, tuple[float, float]]:
    _height_as_x, width_as_y = block_page_size(blocks)
    if side == "left":
        names = ["year", "volume", "author", "title"]
        aliases = {
            "year": {"year"},
            "volume": {"volume", "vol"},
            "author": {"author", "authors"},
            "title": {"title"},
        }
        fallback_centers = {
            "year": 0.08 * width_as_y,
            "volume": 0.16 * width_as_y,
            "author": 0.31 * width_as_y,
            "title": 0.82 * width_as_y,
        }
    else:
        names = ["remark", "status", "subject", "page", "year"]
        aliases = {
            "remark": {"remark", "remarks"},
            "status": {"status"},
            "subject": {"subject", "subjects"},
            "page": {"page", "pages"},
            "year": {"year"},
        }
        fallback_centers = {
            "remark": 0.31 * width_as_y,
            "status": 0.40 * width_as_y,
            "subject": 0.63 * width_as_y,
            "page": 0.88 * width_as_y,
            "year": 0.97 * width_as_y,
        }

    centers: dict[str, float] = {}
    for name in names:
        matches = [
            block
            for block in blocks
            if clean_token(block.text).lower().rstrip(".") in aliases[name]
            and block.xc < block_page_size(blocks)[0] * 0.45
        ]
        if matches:
            centers[name] = sorted(matches, key=lambda block: block.xc)[0].yc
        else:
            centers[name] = fallback_centers[name]

    ordered = sorted(centers.items(), key=lambda item: item[1])
    bounds: dict[str, tuple[float, float]] = {}
    for idx, (name, center) in enumerate(ordered):
        left = 0 if idx == 0 else (ordered[idx - 1][1] + center) / 2
        right = width_as_y if idx == len(ordered) - 1 else (center + ordered[idx + 1][1]) / 2
        bounds[name] = (left, right)
    return bounds


def group_blocks_by_record(blocks: list[Block], body_start_x: float, x_gap: float = 35.0) -> list[list[Block]]:
    body = [block for block in blocks if block.xc > body_start_x and not is_noise_line(block.text)]
    body = sorted(body, key=lambda block: (block.xc, block.yc))
    groups: list[list[Block]] = []
    for block in body:
        if not groups:
            groups.append([block])
            continue
        group_center = sum(item.xc for item in groups[-1]) / len(groups[-1])
        if abs(block.xc - group_center) <= x_gap:
            groups[-1].append(block)
        else:
            groups.append([block])
    return [sorted(group, key=lambda block: block.yc) for group in groups]


def assign_blocks_to_columns(group: list[Block], bounds: dict[str, tuple[float, float]]) -> dict[str, str]:
    values = {name: [] for name in bounds}
    for block in group:
        for name, (left, right) in bounds.items():
            if left <= block.yc < right:
                values[name].append(block.text)
                break
    return {name: clean_cell(" ".join(parts)) for name, parts in values.items()}


def parse_side_blocks(blocks: list[Block], side: str, has_header: bool = False) -> list[dict]:
    if not blocks:
        return []
    bounds = block_column_bounds(blocks, side)
    header_blocks = []
    expected = ["year", "volume", "author", "title"] if side == "left" else ["year", "page", "subject", "status", "remark"]
    for block in blocks:
        if clean_token(block.text).lower().rstrip(".") in set(expected):
            header_blocks.append(block)
    if has_header and header_blocks:
        body_start_x = max(block.xc for block in header_blocks) + 25
    else:
        # Only the first two PDF pages contain column headings. Later pages
        # start directly with records, so trimming by header position would
        # remove the first rows.
        body_start_x = min(block.xc for block in blocks) - 5

    rows: list[dict] = []
    for group in group_blocks_by_record(blocks, body_start_x):
        row = assign_blocks_to_columns(group, bounds)
        if side == "left":
            out = {
                "year": normalize_year(row.get("year", "")),
                "volume": clean_cell(row.get("volume", "")),
                "author": clean_cell(row.get("author", "")),
                "title": clean_cell(row.get("title", "")),
                "_review_reason": "",
            }
            if not is_year_line(out["year"]):
                out["_review_reason"] = "missing or unusual year"
            rows.append(out)
        else:
            out = {
                "year_right": normalize_year(row.get("year", "")),
                "page": clean_cell(row.get("page", "")),
                "subject": clean_cell(row.get("subject", "")),
                "status": clean_cell(row.get("status", "")),
                "remark": clean_cell(row.get("remark", "")),
                "_review_reason": "",
            }
            if out["status"] and not is_status_line(out["status"]):
                out["_review_reason"] = "unusual status"
            rows.append(out)
    return rows


def drop_intro_and_headers(lines: list[str], side: str) -> list[str]:
    if side == "left":
        header = ["title", "author", "volume", "year"]
    else:
        header = ["year", "page", "subject", "status", "remark"]
    lower = [line.lower().rstrip(".") for line in lines]
    for idx in range(len(lower)):
        if lower[idx : idx + len(header)] == header:
            return lines[idx + len(header) :]
    # Some pages omit repeated headers. For right pages, data normally begins at
    # the first year. For left pages, keep the page unless a known title header exists.
    if side == "right":
        for idx, line in enumerate(lines):
            if is_year_line(line):
                return lines[idx:]
    return lines


def parse_left_text(text: str) -> list[dict]:
    lines = drop_intro_and_headers(clean_ocr_lines(text), "left")
    rows: list[dict] = []
    buffer: list[str] = []

    for line in lines:
        if is_noise_line(line):
            continue
        buffer.append(line)
        if not is_year_line(line):
            continue

        year = buffer[-1]
        volume_idx = None
        for idx in range(len(buffer) - 2, -1, -1):
            if is_volume_line(buffer[idx]):
                volume_idx = idx
                break

        if volume_idx is None or volume_idx == 0:
            rows.append(
                {
                    "year": year,
                    "volume": "",
                    "author": "",
                    "title": clean_cell(" ".join(buffer[:-1])),
                    "_review_reason": "could not locate volume/author boundary",
                }
            )
            buffer = []
            continue

        author_idx = volume_idx - 1
        rows.append(
            {
                "year": year,
                "volume": clean_cell(buffer[volume_idx]),
                "author": clean_cell(buffer[author_idx]),
                "title": clean_cell(" ".join(buffer[:author_idx])),
                "_review_reason": "",
            }
        )
        buffer = []

    if buffer:
        rows.append(
            {
                "year": "",
                "volume": "",
                "author": "",
                "title": clean_cell(" ".join(buffer)),
                "_review_reason": "left-page trailing text without year",
            }
        )
    return rows


def parse_right_text(text: str) -> list[dict]:
    lines = drop_intro_and_headers(clean_ocr_lines(text), "right")
    groups: list[list[str]] = []
    current: list[str] = []

    for line in lines:
        if is_noise_line(line):
            continue
        if is_year_line(line):
            if current:
                groups.append(current)
            current = [line]
        elif current:
            current.append(line)
    if current:
        groups.append(current)

    rows: list[dict] = []
    for group in groups:
        year = group[0]
        rest = group[1:]
        page = ""
        if rest and is_page_line(rest[0]):
            page = rest.pop(0)

        status_idx = None
        for idx, line in enumerate(rest):
            if is_status_line(line):
                status_idx = idx
                break

        if status_idx is None:
            subject_parts = rest
            status = ""
            remark_parts: list[str] = []
            reason = "could not locate status"
        else:
            subject_parts = rest[:status_idx]
            status = rest[status_idx]
            remark_parts = rest[status_idx + 1 :]
            reason = ""

        rows.append(
            {
                "year_right": year,
                "page": clean_cell(page),
                "subject": clean_cell(" ".join(subject_parts)),
                "status": clean_cell(status),
                "remark": clean_cell(" ".join(remark_parts)),
                "_review_reason": reason,
            }
        )
    return rows


def merge_pair_by_order(left_rows: list[dict], right_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    total = max(len(left_rows), len(right_rows))
    for idx in range(total):
        left = left_rows[idx] if idx < len(left_rows) else {}
        right = right_rows[idx] if idx < len(right_rows) else {}
        reasons = []
        for source in (left, right):
            if source.get("_review_reason"):
                reasons.append(source["_review_reason"])
        if not left:
            reasons.append("missing left-side record")
        if not right:
            reasons.append("missing right-side record")
        if left.get("year") and right.get("year_right") and left.get("year") != right.get("year_right"):
            reasons.append(f"left/right year mismatch: {left.get('year')} vs {right.get('year_right')}")

        rows.append(
            {
                "year": normalize_year(left.get("year", "") or right.get("year_right", "")),
                "volume": clean_cell(left.get("volume", "")),
                "author": clean_cell(left.get("author", "")),
                "title": clean_cell(left.get("title", "")),
                "page": clean_cell(right.get("page", "")),
                "subject": clean_cell(right.get("subject", "")),
                "status": clean_cell(right.get("status", "")),
                "remark": clean_cell(right.get("remark", "")),
                "_review_reason": "; ".join(dict.fromkeys(reasons)),
            }
        )
    return rows


def parse_side_page(words: list[Word], side: str) -> list[dict]:
    bounds = column_bounds(words, side)
    expected = ["year", "volume", "author", "title"] if side == "left" else ["page", "subject", "status", "remark"]
    header_y = find_header_y(words, expected)
    _width, height = page_size(words)
    top_cut = (header_y + 18) if header_y else height * 0.08
    bottom_cut = height * 0.97
    body_words = [w for w in words if top_cut <= w.yc <= bottom_cut]

    line_rows = []
    for line in group_words_by_line(body_words):
        text = " ".join(w.text for w in line)
        if is_noise_line(text):
            continue
        row = line_to_columns(line, bounds)
        row = {k: clean_cell(v) for k, v in row.items()}
        row["_y"] = sum(w.yc for w in line) / len(line)
        row["_text"] = clean_cell(text)
        line_rows.append(row)

    merged: list[dict] = []
    for row in line_rows:
        if side == "left":
            starts_new = looks_like_year(row.get("year", "")) or (
                looks_like_volume(row.get("volume", "")) and bool(row.get("author") or row.get("title"))
            )
        else:
            starts_new = looks_like_page(row.get("page", "")) or bool(row.get("subject"))

        if starts_new or not merged:
            merged.append(row)
        else:
            for key in expected:
                if row.get(key):
                    sep = " " if merged[-1].get(key) else ""
                    merged[-1][key] = clean_cell(merged[-1].get(key, "") + sep + row[key])
            merged[-1]["_text"] = clean_cell(merged[-1].get("_text", "") + " " + row.get("_text", ""))

    return merged


def nearest_by_y(left_rows: list[dict], right_rows: list[dict], tolerance: float = 28.0) -> list[dict]:
    used_right: set[int] = set()
    output: list[dict] = []

    for left in left_rows:
        ly = float(left.get("_y", 0))
        candidates = [
            (idx, abs(float(right.get("_y", 0)) - ly))
            for idx, right in enumerate(right_rows)
            if idx not in used_right
        ]
        idx = None
        if candidates:
            best_idx, best_gap = min(candidates, key=lambda item: item[1])
            if best_gap <= tolerance:
                idx = best_idx

        right = right_rows[idx] if idx is not None else {}
        if idx is not None:
            used_right.add(idx)

        output.append(
            {
                "year": clean_cell(left.get("year", "")),
                "volume": clean_cell(left.get("volume", "")),
                "author": clean_cell(left.get("author", "")),
                "title": clean_cell(left.get("title", "")),
                "page": clean_cell(right.get("page", "")),
                "subject": clean_cell(right.get("subject", "")),
                "status": clean_cell(right.get("status", "")),
                "remark": clean_cell(right.get("remark", "")),
                "_left_y": ly,
                "_right_y": right.get("_y", ""),
                "_match_gap": "" if idx is None else abs(float(right.get("_y", 0)) - ly),
                "_review_reason": "",
            }
        )

    for idx, right in enumerate(right_rows):
        if idx in used_right:
            continue
        output.append(
            {
                "year": "",
                "volume": "",
                "author": "",
                "title": "",
                "page": clean_cell(right.get("page", "")),
                "subject": clean_cell(right.get("subject", "")),
                "status": clean_cell(right.get("status", "")),
                "remark": clean_cell(right.get("remark", "")),
                "_left_y": "",
                "_right_y": right.get("_y", ""),
                "_match_gap": "",
                "_review_reason": "unmatched right-side row",
            }
        )

    return output


def add_review_flags(rows: list[dict]) -> list[dict]:
    for row in rows:
        reasons = []
        if row.get("_review_reason"):
            reasons.append(row["_review_reason"])
        if not looks_like_year(row.get("year", "")):
            reasons.append("missing or unusual year")
        if not row.get("title"):
            reasons.append("missing title")
        if not row.get("author"):
            reasons.append("missing author")
        if row.get("_match_gap") not in {"", None}:
            try:
                if float(row["_match_gap"]) > 18:
                    reasons.append("left/right vertical match is loose")
            except ValueError:
                pass
        row["_review_reason"] = "; ".join(dict.fromkeys(reasons))
    return rows


def parse_cached_ocr(cache_dir: Path) -> pd.DataFrame:
    cache_files = sorted(cache_dir.glob("page_*.json"))
    if not cache_files:
        raise FileNotFoundError(f"No OCR cache files found in {cache_dir}")

    page_count = len(cache_files)
    all_rows: list[dict] = []
    for left_page in range(1, page_count + 1, 2):
        right_page = left_page + 1
        left_rows = parse_side_blocks(load_page_blocks(cache_dir, left_page), "left", has_header=(left_page == 1))
        right_rows = (
            parse_side_blocks(load_page_blocks(cache_dir, right_page), "right", has_header=(right_page == 2))
            if right_page <= page_count
            else []
        )
        if not left_rows:
            left_rows = parse_left_text(load_page_text(cache_dir, left_page))
        if right_page <= page_count and not right_rows:
            right_rows = parse_right_text(load_page_text(cache_dir, right_page))
        pair_rows = merge_pair_by_order(left_rows, right_rows)
        for row in pair_rows:
            row["_pdf_left_page"] = left_page
            row["_pdf_right_page"] = right_page if right_page <= page_count else ""
        all_rows.extend(pair_rows)

    add_review_flags(all_rows)
    return pd.DataFrame(all_rows)


def write_outputs(df: pd.DataFrame, out_csv: Path, review_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df[OUTPUT_COLUMNS].to_csv(out_csv, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)

    debug_cols = ["_pdf_left_page", "_pdf_right_page", "_left_y", "_right_y", "_match_gap", "_review_reason"]
    review_cols = OUTPUT_COLUMNS + [col for col in debug_cols if col in df.columns]
    review_df = df[df["_review_reason"].astype(str).str.len() > 0].copy()
    review_df[review_cols].to_csv(review_csv, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)


def create_searchable_pdf(pdf_path: Path, cache_dir: Path, out_pdf: Path) -> None:
    """Create a searchable PDF by adding an invisible OCR text layer."""
    doc = fitz.open(pdf_path)
    for page_index, page in enumerate(doc):
        cache_path = cache_dir / f"page_{page_index + 1:03d}.json"
        if not cache_path.exists():
            raise FileNotFoundError(f"Missing OCR cache for PDF page {page_index + 1}: {cache_path}")

        data = json.loads(cache_path.read_text(encoding="utf-8"))
        text = data.get("fullTextAnnotation", {}).get("text", "")
        if not text.strip():
            continue

        x = 1.0
        y = 2.0
        line_height = 1.4
        column_width = 70.0
        for line in clean_ocr_lines(text):
            if y > page.rect.height - 2:
                y = 2.0
                x += column_width
            if x > page.rect.width - 2:
                x = 1.0
                y = 2.0
            page.insert_text(
                fitz.Point(x, y),
                line,
                fontsize=1,
                fontname="helv",
                render_mode=3,
                overlay=True,
            )
            y += line_height

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    if out_pdf.exists():
        out_pdf.unlink()
    doc.save(out_pdf, garbage=4, deflate=True)
    doc.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract index001.pdf into CSV using Google Vision OCR.")
    parser.add_argument("--pdf", type=Path, default=PDF_PATH)
    parser.add_argument("--key", type=Path, default=KEY_PATH)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--out", type=Path, default=OUT_CSV)
    parser.add_argument("--review-out", type=Path, default=REVIEW_CSV)
    parser.add_argument("--ocr-pdf", type=Path, default=OCR_PDF)
    parser.add_argument("--limit-pages", type=int, default=None, help="OCR only the first N pages for testing.")
    parser.add_argument("--ocr-only", action="store_true", help="Only run OCR/cache creation.")
    parser.add_argument("--parse-only", action="store_true", help="Only parse existing OCR cache.")
    parser.add_argument("--no-csv", action="store_true", help="Skip CSV output.")
    parser.add_argument("--no-ocr-pdf", action="store_true", help="Skip searchable OCR PDF output.")
    args = parser.parse_args()

    if not args.parse_only:
        ensure_ocr_cache(args.pdf, args.key, args.cache_dir, args.limit_pages)

    if args.ocr_only:
        return

    if not args.no_csv:
        df = parse_cached_ocr(args.cache_dir)
        write_outputs(df, args.out, args.review_out)
        review_count = int(df["_review_reason"].astype(str).str.len().gt(0).sum())
        print(f"Wrote {len(df)} rows: {args.out}")
        print(f"Wrote {review_count} review rows: {args.review_out}")

    if not args.no_ocr_pdf:
        create_searchable_pdf(args.pdf, args.cache_dir, args.ocr_pdf)
        print(f"Wrote searchable OCR PDF: {args.ocr_pdf}")


if __name__ == "__main__":
    main()
