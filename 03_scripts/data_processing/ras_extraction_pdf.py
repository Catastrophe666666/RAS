"""
RAS Journal Preprocessing Pipeline
皇家亚洲学会期刊预处理流水线
=====================================
Human-in-the-loop: operator confirms page offset per volume.
人机协同：操作员逐本确认页码偏移量。

Text extraction uses Google Vision API (DOCUMENT_TEXT_DETECTION).
文本提取使用 Google Vision API（DOCUMENT_TEXT_DETECTION 模式）。
OCR results are cached in ocr_cache/ to avoid re-billing.
OCR 结果缓存至 ocr_cache/，避免重复计费。

Article metadata (title, author, author_title, location, contributor_role)
is auto-extracted from each article's first-page image via Gemini API
(independent key/project from Vision API), then confirmed by the operator.
文章 metadata 通过 Gemini API 从首页图像自动识别后由操作员逐字段确认或修改。
Gemini API 密钥与 Vision API 完全独立，可属于不同的 Google Cloud 项目。

TOC (CONTENTS) scanning has been removed; all entries are entered manually
with Gemini assistance.
目录（CONTENTS）扫描环节已删除，所有条目改为 Gemini 辅助的手动录入。

Article fulltext is saved as individual .txt files in FULLTEXT_DIR.
每篇文章正文单独保存为 txt 文件，存放于 FULLTEXT_DIR 目录。
Miscellaneous sections are saved as txt only, not written to CSV.
Miscellaneous 部分仅保存 txt，不写入 CSV。
"""

import re
import os
import csv
import sys
import base64
import json
import time
import requests
from pathlib import Path
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

import fitz       # PyMuPDF
import pandas as pd


# ─────────────────────────────────────────────────────────────────────
# CONFIGURATION  /  配置
# ─────────────────────────────────────────────────────────────────────

PDF_DIR      = r"C:\RAS PDF\RAS_Journal\RAS_Journal\Final_PDFs"
OUTPUT_CSV   = "ras_articles.csv"
PROGRESS_CSV = "ras_progress.csv"
ERROR_LOG    = "ras_errors.log"

# Directory for individual article fulltext .txt files
# 每篇文章正文 txt 文件的存放目录
FULLTEXT_DIR = "article_texts"

# Files to skip entirely / 完全跳过的文件
EXCLUDE_FILES = {
    "RAS_Library-Catalogue_1894.pdf",
}

# Files to force-reprocess even if already in ras_progress.csv
# 强制重新处理的文件（即使已在进度记录中）
REPROCESS_FILES = {
    "RAS_XXI_3-4_1886.pdf",
}

# CSV column schema / CSV 字段定义
# fulltext is no longer stored in CSV; fulltext_path points to the .txt file.
# fulltext 不再存入 CSV；fulltext_path 指向对应的 txt 文件路径。
CSV_COLUMNS = [
    "article_id",          # e.g. XIX_1884_01
    "source_file",         # original filename / 原始文件名
    "journal_year",        # primary year extracted from filename / 从文件名提取的主年份
    "volume_label",        # Roman numeral volume / 卷号罗马数字
    "roman_numeral",       # Article roman numeral from TOC / 目录中的文章罗马序号
    "title",               # article title / 文章标题
    "author",              # author name, titles stripped / 作者姓名（头衔已分离）
    "author_title",        # honorifics/titles, e.g. Rev., Esq., H.B.M.'s Consul / 作者头衔
    "location",            # spatial/geographic location / 空间或地理位置
    "contributor_role",    # author / translator / translator_annotator / editor / unknown
    "section_type",        # article / proceedings / other
    "page_start_printed",  # printed page number (start) / 印刷页码（起始）
    "page_end_printed",    # printed page number (end)   / 印刷页码（结束）
    "page_start_scan",     # PDF physical page, 1-based / PDF物理页码（起始，从1计）
    "page_end_scan",       # PDF physical page, 1-based / PDF物理页码（结束，从1计）
    "fulltext_path",       # path to the .txt file containing article fulltext
                           # 包含文章正文的 txt 文件路径
    "char_count",          # character count of fulltext / 全文字符数
    "parse_flag",          # OK / WARNING / SKIPPED / MANUAL
    "parse_notes",         # free-text notes on any issues / 问题说明
    "processed_at",        # ISO timestamp / 处理时间戳
]


# ─────────────────────────────────────────────────────────────────────
# NETWORK ROBUSTNESS SETUP  /  网络高可用与重试机制配置
# ─────────────────────────────────────────────────────────────────────

# Setup robust requests Session with automatic backoff retries to prevent connection drops
# 配置健壮的 requests 会话，带自动指数退避重试功能，防止网络连接意外中断
http_session = requests.Session()
retry_strategy = Retry(
    total=5,                        # Max retries / 最大重试次数
    backoff_factor=1.5,             # Waiting factor (1.5s, 3s, 6s...) / 退避等待系数
    status_forcelist=[429, 500, 502, 503, 504], # Trigger retry on these statuses / 触发重试的状态码
    raise_on_status=False
)
http_adapter = HTTPAdapter(max_retries=retry_strategy)
http_session.mount("https://", http_adapter)
http_session.mount("http://", http_adapter)


# ─────────────────────────────────────────────────────────────────────
# GEMINI API CONFIGURATION  /  Gemini API 配置
# ─────────────────────────────────────────────────────────────────────

# Gemini API key — completely independent from the Vision API key.
_GEMINI_KEY_PATH = r"C:\ras_text_analysis\gemini_key.txt"
try:
    with open(_GEMINI_KEY_PATH, "r", encoding="utf-8") as _gf:
        GEMINI_API_KEY = _gf.read().strip()
except FileNotFoundError:
    print(f"WARNING: Gemini API key file not found: {_GEMINI_KEY_PATH}")
    print(f"警告：找不到 Gemini API Key 文件：{_GEMINI_KEY_PATH}")
    print("Gemini auto-recognition will be disabled. / Gemini 自动识别将禁用。")
    GEMINI_API_KEY = ""

# 使用目前最主流且受支持的稳定版基础端点路径，防止 404
GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent"
)

# Prompt sent to Gemini for first-page metadata extraction
_GEMINI_METADATA_PROMPT = """\
You are a metadata extractor for 19th-century academic journal articles.
The image shows the FIRST PAGE of one article from the Journal of the Royal Asiatic Society.

Extract ONLY the following fields. If a field is not clearly visible, return an empty string for it.
Respond in STRICT JSON with exactly these keys (no extra keys, no markdown fences):

{
  "title": "<full article title as printed>",
  "author": "<author name only, no honorifics or titles>",
  "author_title": "<honorifics / titles, e.g. Rev., Esq., F.R.G.S., H.B.M.'s Consul>",
  "affiliation_location": "<institution or geographic location stated near author name, if any>",
  "contributor_role": "<one of: author | translator | translator_annotator | editor | unknown>"
}

Rules:
- "title" is the article heading (may span multiple lines).
- "author" is the personal name only (e.g. "Thomas Wade"), without any titles or honorifics.
- "author_title" captures everything stripped from the author field (e.g. "Sir, K.C.B., F.R.G.S.").
- "affiliation_location" is a place or institution printed near the author, e.g. "Bengal", "British Consulate, Shanghai".
- "contributor_role": choose translator_annotator if "translated … with notes", translator if "translated by", editor if "edited by", author otherwise.
- Return ONLY the JSON object. No explanation, no markdown.
"""


def gemini_extract_first_page(doc: fitz.Document, pdf_name: str,
                               scan_page: int) -> dict:
    """
    Render `scan_page` of the PDF and call Gemini to extract article metadata.
    """
    _empty = {
        "title": "", "author": "", "author_title": "",
        "affiliation_location": "", "contributor_role": "",
    }

    if not GEMINI_API_KEY:
        return _empty

    total = len(doc)
    if not (1 <= scan_page <= total):
        print(f"    [Gemini] scan_page {scan_page} out of range, skipping.")
        return _empty

    # Render page to PNG / 渲染页面为 PNG
    page     = doc[scan_page - 1]
    mat      = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
    pix      = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
    img_b64  = base64.b64encode(pix.tobytes("png")).decode("utf-8")

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": img_b64,
                        }
                    },
                    {"text": _GEMINI_METADATA_PROMPT},
                ]
            }
        ]
    }

    raw_text = ""
    try:
        # 使用 params 的安全传参形式，杜绝网络传输产生 404
        response = http_session.post(
            GEMINI_API_URL,
            params={"key": GEMINI_API_KEY},
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()

        raw_text = (
            result.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
            .strip()
        )

        # Strip possible markdown fences / 去除可能的 markdown 代码块
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```[a-z]*\n?", "", raw_text)
            raw_text = re.sub(r"\n?```$", "", raw_text)

        extracted = json.loads(raw_text)
        for k in _empty:
            extracted.setdefault(k, "")
        return extracted

    except (requests.RequestException, json.JSONDecodeError, KeyError,
            IndexError) as exc:
        print(f"    [Gemini API error: {exc}]")
        return _empty
    finally:
        time.sleep(1.2)


# ─────────────────────────────────────────────────────────────────────
# FILENAME PARSING  /  文件名解析
# ─────────────────────────────────────────────────────────────────────

def parse_filename(filename: str) -> dict:
    """
    Extract year and volume label from filename.
    """
    stem = Path(filename).stem

    year_match = re.search(r'(\d{4})', stem)
    year = int(year_match.group(1)) if year_match else None

    vol_match = re.match(r'RAS_([IVXLCivxlc]+)', stem)
    volume = vol_match.group(1).upper() if vol_match else "UNKNOWN"

    return {"journal_year": year, "volume_label": volume}


# ─────────────────────────────────────────────────────────────────────
# TEXT EXTRACTION VIA GOOGLE VISION API  /  文本提取（Google Vision API）
# ─────────────────────────────────────────────────────────────────────

# Read Vision API key from key1.txt inside the same folder
# 读取保存在相同目录下的 key1.txt 密钥文件
_API_KEY_PATH = r"C:\ras_text_analysis\key1.txt"
try:
    with open(_API_KEY_PATH, "r", encoding="utf-8") as _f:
        VISION_API_KEY = _f.read().strip()
except FileNotFoundError:
    print(f"ERROR: API key file not found: {_API_KEY_PATH}")
    print(f"错误：找不到 API Key 文件：{_API_KEY_PATH}")
    sys.exit(1)

# 直接构建 Vision API URL，无需正则清洗
VISION_API_URL = f"https://vision.googleapis.com/v1/images:annotate?key={VISION_API_KEY}"

# OCR cache directory / OCR 缓存目录
OCR_CACHE_DIR = "ocr_cache"
os.makedirs(OCR_CACHE_DIR, exist_ok=True)

VISION_API_URL = f"https://vision.googleapis.com/v1/images:annotate?key={VISION_API_KEY}"

# 启动时校验 key 非空
if not VISION_API_KEY:
    print("ERROR: Vision API key is empty. Check key1.txt.")
    sys.exit(1)
    
# Image resolution for rendering PDF pages before sending to Vision API
RENDER_DPI = 200


def _cache_path(pdf_name: str, page_num: int) -> str:
    """
    Return the local cache file path for one OCR result.
    """
    stem = Path(pdf_name).stem
    return os.path.join(OCR_CACHE_DIR, f"{stem}_p{page_num:04d}.txt")


def ocr_page(doc: fitz.Document, pdf_name: str, page_num: int) -> str:
    """
    OCR one PDF page via Google Vision API.
    """
    cache = _cache_path(pdf_name, page_num)

    # Return cached result if available / 有缓存则直接返回
    if os.path.exists(cache):
        return Path(cache).read_text(encoding="utf-8")

    # Render PDF page to image / 将 PDF 页面渲染为图片
    page     = doc[page_num - 1]          # fitz is 0-based
    mat      = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
    pix      = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
    img_bytes = pix.tobytes("png")
    img_b64   = base64.b64encode(img_bytes).decode("utf-8")

    # Call Vision API / 调用 Vision API
    payload = {
        "requests": [
            {
                "image":    {"content": img_b64},
                "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
            }
        ]
    }

    text = ""
    try:
        # 使用过滤好、不带 [https 的干净 URL 进行网络请求
        response = http_session.post(
            VISION_API_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()

        # Extract full text annotation / 提取全文注释
        text = (
            result["responses"][0]
            .get("fullTextAnnotation", {})
            .get("text", "")
        )
    except requests.RequestException as e:
        print(f"    [Vision API error on page {page_num}: {e}]")
        return ""
    finally:
        # Rate-limiting guard: only pauses when a live request happened / 防拉黑限速
        time.sleep(1.2)

    if text.strip():
        with open(cache, "w", encoding="utf-8") as f:
            f.write(text)

    return text


def extract_pages_text(doc: fitz.Document, pdf_name: str,
                       scan_start: int, scan_end: int) -> str:
    """
    OCR and concatenate text from a range of physical PDF pages.
    """
    parts = []
    total = len(doc)
    for page_num in range(scan_start, scan_end + 1):
        if 1 <= page_num <= total:
            text = ocr_page(doc, pdf_name, page_num)
            if text.strip():
                parts.append(text.strip())
    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────
# AUTHOR / TITLE SPLITTING  /  作者姓名与头衔分离
# ─────────────────────────────────────────────────────────────────────

_PREFIX_TITLES = re.compile(
    r'^('
    r'The\s+)?'
    r'(?:'
    r'Right\s+Rev(?:d|erend)?\.?'
    r'|Rev(?:d|erend)?\.?'
    r'|Lt\.?-?Col\.?'
    r'|Col\.?'
    r'|Maj(?:or)?\.?'
    r'|Capt(?:ain)?\.?'
    r'|Dr\.?'
    r'|Prof(?:essor)?\.?'
    r'|Sir'
    r'|Lord'
    r')'
    r'\s+',
    re.IGNORECASE
)

_SUFFIX_TITLES = re.compile(
    r',?\s*('
    r'Esq(?:uire)?\.?'
    r'|(?:of\s+)?H\.?\s*B\.?\s*M\.?\'s\s+(?:Acting\s+)?(?:Consul(?:-General)?|Minister|Resident)'
    r'|of\s+H\.?\s*B\.?\s*M\.?\'s\s+Consular\s+Service'
    r'|F\.?\s*R\.?\s*G\.?\s*S\.?'
    r'|F\.?\s*R\.?\s*A\.?\s*S\.?'
    r'|M\.?\s*R\.?\s*A\.?\s*S\.?'
    r'|C\.?\s*[A-Z]\.?\s*[A-Z]\.?'
    r')'
    r'[\s,]*',
    re.IGNORECASE
)


def split_author_title(raw_author: str) -> tuple:
    """
    Separate honorific titles from an author name string.
    """
    if not raw_author:
        return "", ""

    text = raw_author.strip()
    titles = []

    while True:
        m = _PREFIX_TITLES.match(text)
        if not m:
            break
        titles.append(m.group(0).strip())
        text = text[m.end():].strip()

    while True:
        m = _SUFFIX_TITLES.search(text)
        if not m:
            break
        titles.append(m.group(1).strip())
        text = (text[:m.start()] + text[m.end():]).strip().rstrip(',').strip()

    name        = text
    title_str   = ", ".join(titles)
    return name, title_str


# ─────────────────────────────────────────────────────────────────────
# HUMAN-IN-THE-LOOP INPUT  /  人机交互输入
# ─────────────────────────────────────────────────────────────────────

def prompt_int(prompt_en: str, prompt_zh: str, allow_zero: bool = False, allow_empty: bool = False):
    full_prompt = f"  {prompt_en} / {prompt_zh}: "
    while True:
        raw = input(full_prompt).strip()
        if raw.lower() == 'q':
            return None
        if allow_empty and raw == '':
            return ""
        if allow_zero and raw == '0':
            return 0
        try:
            val = int(raw)
            if val > 0 or (allow_zero and val == 0):
                return val
        except ValueError:
            pass
        print("  [Invalid input. Enter a positive integer, or 'q' to quit.]")
        print("  [输入无效，请输入正整数，或输入 q 退出。]")


def enter_article_metadata(pdf_name: str,
                           doc: fitz.Document = None) -> tuple:
    """
    Directly prompt the operator to enter article metadata for each article,
    with Gemini API auto-recognition of the first-page image as pre-fill.
    """
    ROLE_OPTIONS = {
        "1": "author",
        "2": "translator",
        "3": "translator_annotator",
        "4": "editor",
        "5": "unknown",
    }

    print()
    print("  [Step 2 / 第二步]  Article metadata entry with Gemini assistance")
    print("  [Step 2 / 第二步]  逐篇录入文章 metadata（Gemini 辅助识别）")
    print()

    # ── Choose page number mode ────────────────────────────────────────
    print("  Which page numbers will you enter? / 你将输入哪种页码？")
    print("    1 = Printed page numbers  印刷页码")
    print("    2 = Physical page numbers  物理页码")
    print()
    while True:
        mode_raw = input("  Enter 1 or 2 / 输入 1 或 2: ").strip()
        if mode_raw in ("1", "2"):
            break
        print("  [Invalid. Enter 1 or 2. / 输入无效，请输入 1 或 2。]")

    page_input_mode = "printed" if mode_raw == "1" else "physical"
    page_label_en   = "Physical" if page_input_mode == "physical" else "Printed"
    page_label_zh   = "物理"     if page_input_mode == "physical" else "印刷"

    # ── Offset hint for Gemini scan-page calculation (printed mode only) ──
    offset_hint = 0
    if doc is not None and GEMINI_API_KEY and page_input_mode == "printed":
        print()
        print("  For Gemini to locate the correct first-page image, enter the offset")
        print("  (= physical page number where printed page 1 appears, minus 1).")
        print("  If unknown yet, enter 0.")
        print()
        print("  为使 Gemini 正确定位首页图像，请输入页码偏移量")
        print("  （= 印刷第 1 页对应的物理页码 − 1）。若暂不清楚，输入 0。")
        _off_raw = input("  Offset / 偏移量 [default 0]: ").strip()
        try:
            offset_hint = int(_off_raw)
        except ValueError:
            offset_hint = 0

    print()

    # ── Article-by-article input loop ─────────────────────────────────
    entries = []
    idx = 1
    while True:
        p = prompt_int(
            f"Article {idx} {page_label_en} start page  "
            f"[0 = done, {idx-1} entered so far]",
            f"第 {idx} 篇{page_label_zh}起始页  "
            f"【输入 0 = 录入完毕，已录 {idx-1} 篇】",
            allow_zero=True,
        )
        if p == 0 or p is None:
            print()
            print(f"  Entry complete. {len(entries)} article(s) recorded.")
            print(f"  录入完毕，共 {len(entries)} 篇。")
            break

        # ── Gemini first-page recognition (per-article opt-in) ──────────
        # Ask each time so operator can skip Gemini for individual articles.
        # 每篇单独询问，操作员可逐篇决定是否调用 Gemini。
        ai = {}
        if doc is not None and GEMINI_API_KEY:
            _use = input(
                f"  Art.{idx} — Use Gemini? / 是否启用 Gemini？ [y / n]: "
            ).strip().lower()
            if _use == "y":
                scan_page = p if page_input_mode == "physical" else p + offset_hint
                print(f"  [Gemini] Analysing first-page image (scan p.{scan_page}) ...")
                print(f"  [Gemini] 正在识别首页图像（扫描页 {scan_page}）……")
                ai = gemini_extract_first_page(doc, pdf_name, scan_page)

            if any(ai.values()):
                print()
                print("  ┌── Gemini recognition / Gemini 识别结果 " + "─" * 30)
                print(f"  │  title            : {ai.get('title','')}")
                print(f"  │  author           : {ai.get('author','')}")
                print(f"  │  author_title     : {ai.get('author_title','')}")
                print(f"  │  affiliation/loc  : {ai.get('affiliation_location','')}")
                print(f"  │  contributor_role : {ai.get('contributor_role','')}")
                print("  └── Press Enter to accept each field, or type to override ──")
                print("  └── 每字段直接回车接受，或输入内容覆盖 ───────────────────")
                print()
            else:
                print("  [Gemini] No result — please enter manually.")
                print("  [Gemini] 未返回结果，请手动输入。")
                print()

        def _ask(field_en, field_zh, ai_key):
            """Show Gemini suggestion in prompt; Enter = accept."""
            suggestion = ai.get(ai_key, "") if ai else ""
            if suggestion:
                prompt = f"  Art.{idx} {field_en} [{suggestion}] (↵=accept): "
            else:
                prompt = f"  Art.{idx} {field_en} (↵ to skip): "
            val = input(prompt).strip()
            return suggestion if (val == "" and suggestion) else val

        title        = _ask("title",       "文章标题",  "title")
        author       = _ask("author",       "作者姓名",  "author")
        author_title = _ask("author title", "作者头衔",  "author_title")
        location     = _ask("location",     "地点/机构", "affiliation_location")

        # Contributor role
        ai_role = ai.get("contributor_role", "") if ai else ""
        role_prompt = (
            f"  Art.{idx} role (1-5, ↵=accept '{ai_role}'): "
            if ai_role else
            f"  Art.{idx} role (1-5, ↵=author): "
        )
        print("    1=author  2=translator  3=translator_annotator  "
              "4=editor  5=unknown")
        while True:
            role_raw = input(role_prompt).strip()
            if role_raw == "":
                contributor_role = (
                    ai_role if ai_role in ROLE_OPTIONS.values() else "author"
                )
                break
            if role_raw in ROLE_OPTIONS:
                contributor_role = ROLE_OPTIONS[role_raw]
                break
            print("  [Invalid. Enter 1-5 or press Enter. / 输入无效，请输入 1-5 或回车。]")

        # ── Optional manual end page ───────────────────────────────────
        # 可选：手动录入结束页（留空则由下一篇起始页自动推算）
        end_p_raw = input(
            f"  Art.{idx} {page_label_en} end page  "
            f"(↵ = auto from next article start / 留空由下篇起始自动推算): "
        ).strip()
        page_end_manual = None
        if end_p_raw:
            try:
                page_end_manual = int(end_p_raw)
                if page_end_manual < p:
                    print(f"  [Warning: end page {page_end_manual} < start page {p}. Value kept as entered.]")
                    print(f"  [警告：结束页 {page_end_manual} 小于起始页 {p}，已按输入值保存。]")
            except ValueError:
                print("  [Invalid — end page ignored, will auto-infer. / 输入无效，忽略，将自动推算。]")
                page_end_manual = None

        entry_dict = {
            "roman_numeral":    str(idx),
            "title":            title,
            "author":           author,
            "author_title":     author_title,
            "location":         location,
            "contributor_role": contributor_role,
            "page_printed":     str(p),
            "section_type":     "article",
        }
        if page_end_manual is not None:
            entry_dict["page_end_manual"] = page_end_manual

        entries.append(entry_dict)
        idx += 1

    # ── Optional Miscellaneous section ────────────────────────────────
    print()
    print("  Does this volume have a Miscellaneous section?")
    print("  本卷是否包含 Miscellaneous 部分？")
    has_misc = input("  y / n: ").strip().lower()

    if has_misc == "y":
        print()
        misc_start = prompt_int(
            f"Miscellaneous {page_label_en} start page",
            f"Miscellaneous {page_label_zh}起始页",
        )
        if misc_start is None:
            return entries, page_input_mode

        misc_end = prompt_int(
            f"Miscellaneous {page_label_en} end page",
            f"Miscellaneous {page_label_zh}结束页",
        )
        if misc_end is None:
            return entries, page_input_mode

        entries.append({
            "roman_numeral":    "",
            "title":            "Miscellaneous",
            "author":           "",
            "author_title":     "",
            "location":         "",
            "contributor_role": "unknown",
            "page_printed":     str(misc_start),
            "page_end_manual":  misc_end,
            "section_type":     "other",
        })
        print(f"  Miscellaneous recorded: pages {misc_start}–{misc_end}.")
        print(f"  Miscellaneous 已记录：页码 {misc_start}–{misc_end}。")

    return entries, page_input_mode


# ─────────────────────────────────────────────────────────────────────
# PROGRESS TRACKING  /  断点续跑
# ─────────────────────────────────────────────────────────────────────

def load_completed_files(progress_path: str) -> set:
    """
    Read ras_progress.csv and return the set of already-processed filenames.
    Handles BOM (utf-8-sig) headers, normalises column names, and strips
    whitespace so the comparison against all_pdfs never silently fails.
    已完成文件名集合读取：兼容 BOM 头、列名空白、编码差异，防止静默空集合。
    """
    if not os.path.exists(progress_path):
        return set()
    if os.path.getsize(progress_path) == 0:
        return set()
    try:
        # utf-8-sig strips the BOM automatically; works on plain utf-8 too
        df = pd.read_csv(progress_path, encoding="utf-8-sig", dtype=str)
        # Normalise column names: strip whitespace and any residual BOM chars
        df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
        if "source_file" not in df.columns:
            print(f"  WARNING: '{progress_path}' has no 'source_file' column.")
            print(f"  警告：进度文件缺少 'source_file' 列，实际列名：{list(df.columns)}")
            return set()
        return set(df["source_file"].dropna().str.strip().tolist())
    except Exception as exc:
        print(f"  WARNING: Could not read progress file '{progress_path}': {exc}")
        print(f"  警告：无法读取进度文件，将从头处理所有文件。")
        return set()


def mark_file_complete(progress_path: str, filename: str):
    """
    Append one completion record to ras_progress.csv.
    Uses utf-8-sig consistently so the file can be opened in Excel without
    column-name corruption, and so load_completed_files always reads it back
    correctly.
    写入时统一使用 utf-8-sig，与读取端保持一致，避免 BOM 造成列名偏移。
    """
    write_header = not os.path.exists(progress_path) or os.path.getsize(progress_path) == 0
    with open(progress_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["source_file", "completed_at"])
        writer.writerow([filename, datetime.now().isoformat()])


# ─────────────────────────────────────────────────────────────────────
# CSV WRITING  /  CSV 写入
# ─────────────────────────────────────────────────────────────────────

def init_csv(output_path: str):
    if not os.path.exists(output_path):
        with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()


def append_rows(output_path: str, rows: list):
    with open(output_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writerows(rows)


# ─────────────────────────────────────────────────────────────────────
# PER-FILE PROCESSING  /  单文件处理
# ─────────────────────────────────────────────────────────────────────

def process_pdf(pdf_path: str, output_csv: str, progress_csv: str):
    pdf_name  = Path(pdf_path).name
    file_meta = parse_filename(pdf_name)

    print()
    print("=" * 65)
    print(f"  File / 文件    : {pdf_name}")
    print(f"  Year / 年份    : {file_meta['journal_year']}")
    print(f"  Volume / 卷号  : {file_meta['volume_label']}")
    print("=" * 65)

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f"  Total PDF pages / PDF 总页数: {total_pages}")

    print()
    print("  [Step 2 / 第二步]  Entering article metadata (Gemini-assisted) ...")
    entries, page_input_mode = enter_article_metadata(pdf_name, doc=doc)

    if not entries:
        print("  No entries confirmed. Skipping this file.")
        doc.close()
        return

    print()
    if page_input_mode == "physical":
        offset = 0
        print("  [Step 3 / 第三步]  Page offset skipped (physical page mode).")
    else:
        print("  [Step 3 / 第三步]  Page offset / 页码偏移量")
        print("  请在 PDF 阅读器中打开此文件，翻至正文印刷第 1 页（Article I 起始处）。")
        print("  记下阅读器显示的物理页码。")
        print()

        scan_of_page1 = prompt_int(
            "Physical page number in viewer where printed page 1 begins",
            "阅读器中印刷第 1 页对应的物理页码",
        )
        if scan_of_page1 is None:
            print("  Quit signal received. Exiting.")
            doc.close()
            sys.exit(0)

        offset = scan_of_page1 - 1
        print(f"  Offset locked / 偏移量确认: {offset}")

    print()
    print("  (Optional) If this volume has trailing ads, member lists, or appendices after final article...")
    last_page_input = prompt_int(
        "Last page of body text (Enter to skip)",
        "正文最后一页页码（回车跳过）",
        allow_empty=True,
    )
    if last_page_input is None:
        doc.close()
        sys.exit(0)

    print()
    print("  [Step 4 / 第四步]  Extracting article texts / 提取文章全文...")

    os.makedirs(FULLTEXT_DIR, exist_ok=True)

    def to_int_page(p_str):
        try:
            return int(p_str)
        except ValueError:
            return -1

    sortable = sorted(
        [(e, to_int_page(e["page_printed"])) for e in entries
         if isinstance(e, dict)],
        key=lambda x: x[1]
    )

    non_dict = [e for e in entries if not isinstance(e, dict)]
    if non_dict:
        print(f"  WARNING: {len(non_dict)} non-dict entry(entries) in entries list, skipped:")
        print(f"  警告：entries 列表中有 {len(non_dict)} 个非字典条目，已跳过：")
        for nd in non_dict:
            print(f"    {repr(nd)}")

    rows = []
    article_counter = 1

    for idx, (entry, p_input_int) in enumerate(sortable):
        
        if not isinstance(entry, dict):
            continue

        if idx + 1 < len(sortable):
            if "page_end_manual" in entry:
                p_end_input = entry["page_end_manual"]
            else:
                p_end_input = sortable[idx + 1][1] - 1
        else:
            if isinstance(last_page_input, int):
                p_end_input = last_page_input
            else:
                p_end_input = total_pages

        if page_input_mode == "physical":
            scan_start      = p_input_int
            scan_end        = p_end_input
            p_start_printed = p_input_int - offset
            p_end_printed   = p_end_input  - offset
        else:
            scan_start      = p_input_int + offset
            scan_end        = p_end_input  + offset
            p_start_printed = p_input_int
            p_end_printed   = p_end_input

        is_misc = (
            entry.get("section_type") == "other"
            and "miscellaneous" in entry.get("title", "").lower()
        )

        if p_input_int < 0:
            print(
                f"    [--]  SKIPPED         "
                f"pre-body '{entry['page_printed']}'  "
                f"{entry['title'][:40]}"
            )
            article_counter += 1
            continue

        parse_flag  = "OK"
        parse_notes = ""

        if scan_start < 1 or scan_start > total_pages:
            parse_flag  = "WARNING"
            parse_notes = f"scan_start {scan_start} out of range [1, {total_pages}]."
        if scan_end > total_pages:
            parse_notes += f" scan_end clamped from {scan_end} to {total_pages}."
            scan_end = total_pages

        fulltext = extract_pages_text(doc, pdf_name, scan_start, scan_end)

        year_str   = str(file_meta["journal_year"]) if file_meta["journal_year"] else "unknown"
        vol_str    = file_meta["volume_label"]

        if is_misc:
            txt_filename = f"{vol_str}_{year_str}_misc.txt"
        else:
            txt_filename = f"{vol_str}_{year_str}_{article_counter:02d}.txt"

        txt_path = os.path.join(FULLTEXT_DIR, txt_filename)

        with open(txt_path, "w", encoding="utf-8") as tf:
            tf.write(fulltext)

        char_count = len(fulltext)

        print(
            f"    [{idx+1:02d}]  "
            f"{'MISC (txt only)' if is_misc else entry['section_type']:<16}  "
            f"printed {str(p_start_printed):>4}-{str(p_end_printed):<4}  "
            f"scan {str(scan_start):>4}-{str(scan_end):<4}  "
            f"{parse_flag:<8}  {entry['title'][:30]}  -> {txt_filename}"
        )

        if is_misc:
            article_counter += 1
            continue

        article_id = f"{vol_str}_{year_str}_{article_counter:02d}"

        row = {
            "article_id":         article_id,
            "source_file":        pdf_name,
            "journal_year":       file_meta["journal_year"],
            "volume_label":       vol_str,
            "roman_numeral":      entry.get("roman_numeral", ""),
            "title":              entry.get("title", ""),
            "author":             entry.get("author", ""),
            "author_title":       entry.get("author_title", ""),
            "location":           entry.get("location", ""),
            "contributor_role":   entry.get("contributor_role", "author"),
            "section_type":       entry.get("section_type", "article"),
            "page_start_printed": p_start_printed,
            "page_end_printed":   p_end_printed,
            "page_start_scan":    scan_start,
            "page_end_scan":      scan_end,
            "fulltext_path":      txt_path,
            "char_count":         char_count,
            "parse_flag":         parse_flag,
            "parse_notes":        parse_notes,
            "processed_at":       datetime.now().isoformat(),
        }
        # ── Immediate per-article CSV flush ──────────────────────────────
        # Write this article row to CSV right away, before the next OCR call.
        # Prevents total data loss if the program exits mid-volume.
        # 逐篇立即写入 CSV，避免程序在文件处理中途退出时丢失整卷数据。
        append_rows(output_csv, [row])
        rows.append(row)
        article_counter += 1

    # Step 5: Finalise / 第五步：收尾
    misc_count = sum(
        1 for e, _ in sortable
        if isinstance(e, dict)
        and e.get("section_type") == "other"
        and "miscellaneous" in e.get("title", "").lower()
        and to_int_page(e["page_printed"]) >= 0
    )

    # Mark volume complete in progress file ONLY after all article rows are
    # confirmed written. If append_rows raised earlier, this line is never
    # reached and the volume will be retried on next run.
    # 仅在所有文章行均已写入 CSV 后才标记该卷为已完成。
    # 若中途写入失败，progress 不更新，下次运行自动重试整卷。
    mark_file_complete(progress_csv, pdf_name)
    doc.close()

    print()
    print(f"  Written {len(rows)} rows to CSV / 已写入 {len(rows)} 行至 CSV  ->  {output_csv}")
    if misc_count:
        print(f"  Miscellaneous sections saved as txt only / "
              f"Miscellaneous 部分仅保存为 txt（共 {misc_count} 个）")


# ─────────────────────────────────────────────────────────────────────
# MAIN  /  主程序
# ─────────────────────────────────────────────────────────────────────

def main():
    pdf_dir = Path(PDF_DIR)
    output_csv = OUTPUT_CSV

    if not pdf_dir.exists():
        print(f"错误：PDF 目录不存在：{pdf_dir}")
        sys.exit(1)

    all_pdfs = sorted(
        f.name for f in pdf_dir.glob("*.pdf")
        if f.name not in EXCLUDE_FILES
    )

    if not all_pdfs:
        print("未找到 PDF 文件，请检查 PDF_DIR 配置。")
        sys.exit(1)

    # ── Diagnostic: show raw progress file content on first load ────────
    # 启动诊断：打印进度文件原始读取情况，便于排查"已完成=0"问题
    if os.path.exists(PROGRESS_CSV) and os.path.getsize(PROGRESS_CSV) > 0:
        try:
            _diag = pd.read_csv(PROGRESS_CSV, encoding="utf-8-sig", dtype=str)
            _diag.columns = [c.strip().lstrip("\ufeff") for c in _diag.columns]
            print(f"  [Diag] Progress file columns   : {list(_diag.columns)}")
            print(f"  [Diag] Progress file row count : {len(_diag)}")
        except Exception as _e:
            print(f"  [Diag] Could not read progress file for diagnostics: {_e}")
    # ─────────────────────────────────────────────────────────────────────

    completed = load_completed_files(PROGRESS_CSV) - REPROCESS_FILES
    remaining = [f for f in all_pdfs if f not in completed]

    print()
    print("RAS Journal Preprocessing Pipeline")
    print("皇家亚洲学会期刊预处理流水线")
    print("-" * 65)
    print(f"  PDF directory / 目录   : {pdf_dir}")
    print(f"  Total files / 文件总数 : {len(all_pdfs)}")
    print(f"  Completed / 已完成     : {len(completed)}")
    print(f"  Remaining / 待处理     : {len(remaining)}")
    print(f"  Output / 输出文件      : {output_csv}")
    print(f"  Fulltext dir / 正文目录: {FULLTEXT_DIR}")
    if REPROCESS_FILES:
        print(f"  Reprocessing / 重新处理: {', '.join(sorted(REPROCESS_FILES))}")
    print("-" * 65)

    if not remaining:
        print("  所有文件均已处理完毕。")
        return

    init_csv(output_csv)

    for pdf_name in remaining:
        pdf_path = str(pdf_dir / pdf_name)
        try:
            process_pdf(pdf_path, output_csv, PROGRESS_CSV)
        except KeyboardInterrupt:
            print()
            print("  操作员中断。进度已保存，下次运行将从断点继续。")
            sys.exit(0)
        except Exception as exc:
            print(f"  错误：处理 {pdf_name} 时发生异常：{exc}")
            print("  Skipping to next file. / 跳过，继续处理下一个文件。")
            with open(ERROR_LOG, "a", encoding="utf-8") as log:
                log.write(f"{datetime.now().isoformat()}  {pdf_name}  {exc}\n")

    print()
    print("=" * 65)
    print("  Pipeline complete. / 全部处理完毕。")
    print("=" * 65)


if __name__ == "__main__":
    main()