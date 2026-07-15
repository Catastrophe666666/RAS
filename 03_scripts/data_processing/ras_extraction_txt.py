"""
RAS Journal Text Processor - Manual Seq-Based Segmentation
RAS期刊文本处理器 - 按seq手动切分文章

Usage / 用法:
    python 17_new_extract.py

Output / 输出:
  Haithtrust_Added.csv                          - metadata table / 元数据表
  fulltext_txt_by_article_id/<article_id>.txt   - one fulltext file per article / 每篇文章一个正文文件
"""

import os, re, json, time, glob, datetime, csv
import warnings
import requests
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# Proxy and SSL bypass settings / 代理与SSL绕过设置
proxy_url = "http://127.0.0.1:7897"
os.environ["http_proxy"] = proxy_url
os.environ["https_proxy"] = proxy_url
os.environ["HTTP_PROXY"] = proxy_url
os.environ["HTTPS_PROXY"] = proxy_url
os.environ["all_proxy"] = proxy_url

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

# Configuration / 配置
TXT_DIR         = r"C:\ras_text_analysis\cut"
GEMINI_KEY_PATH = r"C:\ras_text_analysis\gemini_key.txt"
OUTPUT_CSV      = r"C:\ras_text_analysis\Haithtrust_Added.csv"
OUTPUT_TXT_DIR  = r"C:\ras_text_analysis\fulltext_txt_by_article_id"

GEMINI_MODEL       = "gemini-2.5-flash"
ARTICLE_CHAR_LIMIT = 400
RATE_LIMIT_SLEEP   = 1.0

# Network retry setup / 网络重试设置
http_session = requests.Session()
retry_strategy = Retry(
    total=5,                        
    backoff_factor=1.5,             
    status_forcelist=[429, 500, 502, 503, 504], 
    raise_on_status=False
)
http_adapter = HTTPAdapter(max_retries=retry_strategy)
http_session.mount("https://", http_adapter)
http_session.mount("http://", http_adapter)

# Constants / 常量
COLUMNS = [
    "article_id", "source_file", "htid", "start_year", "end_year", "volume_label", "roman_numeral",
    "title", "author", "author_title", "location", "contributor_role", "section_type",
    "page_start", "page_end", "seq_start", "seq_end", "page_url",
    "fulltext_path", "char_count", "parse_flag", "parse_notes", "processed_at"
]
COL_WIDTHS = {
    "article_id": 18, "source_file": 22, "htid": 26,
    "start_year": 12, "end_year": 12, "volume_label": 13, "roman_numeral": 14,
    "title": 45, "author": 30, "author_title": 14, "location": 18,
    "contributor_role": 18, "section_type": 14,
    "page_start": 11, "page_end": 10, "seq_start": 10, "seq_end": 10,
    "page_url": 55, "fulltext_path": 50, "char_count": 11,
    "parse_flag": 11, "parse_notes": 30, "processed_at": 20
}
HATHI_BASE = "https://babel.hathitrust.org/cgi/pt"
PAGE_RE = re.compile(r'## p\.\s*([^#\(\)]*?)\s*\(#(\d+)\)\s+#+')

SYSTEM_PROMPT = """\
You are a metadata extractor for 19th-century Royal Asiatic Society journal articles.
Given the opening text of one article (raw OCR), extract the fields below and return
ONLY a valid JSON object - no markdown, no commentary.

Fields:
  title            - Article title as printed. Preserve original language characters. (string)
  author           - Author surname(s) and given name/initials as printed (string or null)
  author_title     - Honorific only: e.g. 'DR.', 'REV.', 'ESQ.', 'PROF.' (string or null)
  location         - Geographic location in author affiliation if stated (string or null)
  contributor_role - One of: author | translator | translator_annotator | editor | unknown
  section_type     - One of: article | note | review | address | correspondence | other

Return ONLY the JSON object.
"""

# Helpers / 辅助函数
def parse_file_header(text, filename):
    stem = Path(filename).stem
    htid = None
    m = re.search(r'hdl\.handle\.net/2027/([^\s\n]+)', text)
    if m:
        htid = m.group(1).strip()
    title_m     = re.search(r'^Title:\s+(.+)$',     text, re.MULTILINE)
    publisher_m = re.search(r'^Publisher:\s+(.+)$', text, re.MULTILINE)
    publisher = publisher_m.group(1).strip() if publisher_m else ""
    year = None
    for src in [publisher, stem]:
        ym = re.search(r'(\d{4})', src)
        if ym:
            year = int(ym.group(1))
            break
    roman_m = re.search(r'_(I{1,3}V?|VI{0,3}|IX|X{1,3}|XL|L)(_|$)', stem, re.IGNORECASE)
    roman   = roman_m.group(1).upper() if roman_m else ""
    if not roman:
        vol_m = re.search(r'v_(\d+)', stem, re.IGNORECASE)
        roman = vol_m.group(1) if vol_m else ""
    return {
        "source_file":   stem,
        "htid":          htid or "",
        "journal_title": title_m.group(1).strip() if title_m else "",
        "publisher":     publisher,
        "journal_year":  year or "",
        "volume_label":  roman,
        "roman_numeral": roman,
    }

def infer_end_year(start_year, end_token):
    if len(end_token) == 2:
        end_year = (start_year // 100) * 100 + int(end_token)
        if end_year < start_year:
            end_year += 100
        return end_year
    return int(end_token)

def parse_year_range_from_filename(filename, fallback_year=""):
    stem = Path(filename).stem
    range_m = re.search(r'(?<!\d)((?:16|17|18|19|20)\d{2})\s*[-–—_]\s*(\d{2,4})(?!\d)', stem)
    if range_m:
        start_year = int(range_m.group(1))
        end_year = infer_end_year(start_year, range_m.group(2))
        return start_year, end_year

    years = [int(y) for y in re.findall(r'(?<!\d)((?:16|17|18|19|20)\d{2})(?!\d)', stem)]
    if len(years) >= 2:
        return years[0], years[1]
    if len(years) == 1:
        return years[0], years[0]
    if str(fallback_year).isdigit() and len(str(fallback_year)) == 4:
        year = int(fallback_year)
        return year, year
    return "", ""

def parse_year_range_input(raw_value, default_start="", default_end=""):
    raw_value = raw_value.strip()
    if not raw_value:
        return default_start, default_end

    range_m = re.fullmatch(r'((?:16|17|18|19|20)\d{2})\s*[-–—_]\s*(\d{2,4})', raw_value)
    if range_m:
        start_year = int(range_m.group(1))
        end_year = infer_end_year(start_year, range_m.group(2))
        return start_year, end_year

    single_m = re.fullmatch(r'(?:16|17|18|19|20)\d{2}', raw_value)
    if single_m:
        year = int(raw_value)
        return year, year

    return None, None

def build_page_index(text):
    index = {}
    for m in PAGE_RE.finditer(text):
        seq = int(m.group(2))
        raw_pp = m.group(1).strip() if m.group(1) else ""
        
        # Accept both Arabic page numbers and Roman/front-matter labels. / 兼容阿拉伯数字页码和罗马数字等前置页码。
        if raw_pp.isdigit():
            print_page = int(raw_pp)
        elif raw_pp:
            print_page = raw_pp  # Keep labels such as xix as strings. / xix等页码标签保留为字符串。
        else:
            print_page = None
            
        index[seq] = {"print_page": print_page, "char_pos": m.start()}
    return index

def seq_to_char_pos(seq, index):
    entry = index.get(seq)
    return entry["char_pos"] if entry else None


def gemini_extract(raw_text, api_key, retries=5):
    """Extract metadata via REST API. / 通过REST API提取元数据。"""
    snippet = raw_text[:ARTICLE_CHAR_LIMIT]
    prompt  = SYSTEM_PROMPT + "\n\nARTICLE TEXT:\n" + snippet
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1}
    }

    raw = ""
    for attempt in range(retries):
        try:
            # Use the shared session so proxy and retry settings apply. / 使用共享会话以应用代理和重试设置。
            response = http_session.post(
                url,
                params={"key": api_key},
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=60,
                verify=False
            )
            response.raise_for_status()
            result = response.json()
            
            raw = (
                result.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
                .strip()
            )
            
            backticks = chr(96) * 3
            raw  = re.sub(r'^' + backticks + r'(?:json)?\s*', '', raw, flags=re.IGNORECASE)
            raw  = re.sub(r'\s*' + backticks + r'$', '', raw)
            
            return json.loads(raw)
        except json.JSONDecodeError:
            hit = re.search(r'\{.*\}', raw, re.DOTALL)
            if hit:
                try:
                    return json.loads(hit.group())
                except Exception:
                    pass
        except Exception as e:
            print(f"  [Error / 错误] API call failed (attempt {attempt + 1}/{retries}): {e}. Waiting 10 seconds before retry / 10秒后重试...")
            time.sleep(10)
    return {}


def show_seq_table(page_index, n_head=30):
    seqs = sorted(page_index.keys())
    print(f"  {'seq':>6}  {'print_page':>14}")
    print("  " + "-" * 24)
    for s in seqs[:n_head]:
        pp = page_index[s]["print_page"]
        print(f"  {s:>6}  {str(pp) if pp is not None else '(front matter)':>14}")
    if len(seqs) > n_head:
        print(f"  ... ({len(seqs) - n_head} more -- type 'show all' to see all)")

def show_seq_table_full(page_index):
    seqs = sorted(page_index.keys())
    print(f"  {'seq':>6}  {'print_page':>14}")
    print("  " + "-" * 24)
    for s in seqs:
        pp = page_index[s]["print_page"]
        print(f"  {s:>6}  {str(pp) if pp is not None else '(front matter)':>14}")

# CSV, TXT, and history scan / CSV、TXT与历史扫描
def init_csv(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            old_fields = reader.fieldnames or []
            rows = list(reader)
        if old_fields != COLUMNS:
            if not rows:
                with open(path, "w", encoding="utf-8-sig", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=COLUMNS)
                    writer.writeheader()
                print(f"CSV:  Rebuilt empty metadata file with start_year/end_year / 已用新表头重建空CSV - {path}")
                return
            backup_path = f"{path}.bak_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.replace(path, backup_path)
            migrated_rows = []
            for row in rows:
                if "journal_year" in row and "start_year" not in row:
                    row["start_year"] = row.get("journal_year", "")
                    row["end_year"] = row.get("journal_year", "")
                migrated_rows.append({col: row.get(col, "") for col in COLUMNS})
            with open(path, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=COLUMNS)
                writer.writeheader()
                writer.writerows(migrated_rows)
            print(f"CSV:  Migrated header to start_year/end_year / 已迁移表头，备份: {backup_path}")
            return
        print(f"CSV:  Resuming existing metadata file / 继续写入已有CSV - {path}")
        return
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
    print(f"CSV:  Created new metadata file / 已创建元数据文件 - {path}")

def append_csv_row(path, row_dict):
    with open(path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writerow({col: row_dict.get(col, "") for col in COLUMNS})

def safe_txt_filename(article_id):
    safe_id = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(article_id)).strip()
    return safe_id or "untitled_article"

def write_article_txt(output_dir, article_id, article_text):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{safe_txt_filename(article_id)}.txt")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(article_text)
    return path

def scan_existing_history(csv_path, txt_dir):
    counters = {}
    existing_ids = set()

    if os.path.isdir(txt_dir):
        for txt_path in glob.glob(os.path.join(txt_dir, "*.txt")):
            aid = Path(txt_path).stem
            existing_ids.add(aid)
            m = re.search(r'^(.*?)_((?:\d{4})(?:-\d{4})?)_(\d+)$', str(aid))
            if m:
                vol, year_label, idx = m.group(1), m.group(2), int(m.group(3))
                key = f"{vol}_{year_label}"
                if idx > counters.get(key, 0):
                    counters[key] = idx

    if os.path.exists(csv_path):
        try:
            with open(csv_path, encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    aid = row.get("article_id")
                    if not aid:
                        continue
                    existing_ids.add(aid)
                    m = re.search(r'^(.*?)_((?:\d{4})(?:-\d{4})?)_(\d+)$', str(aid))
                    if m:
                        vol, year_label, idx = m.group(1), m.group(2), int(m.group(3))
                        key = f"{vol}_{year_label}"
                        if idx > counters.get(key, 0):
                            counters[key] = idx
        except Exception as e:
            print(f"  [Warning] CSV history scan issue [CSV history scan issue]: {e}")

    return counters, existing_ids

# Main program / 主程序
def main():
    with open(GEMINI_KEY_PATH) as f:
        api_key = f.read().strip()
        
    print("Gemini: Ready / 已就绪 (Using robust REST API via requests)\n")

    init_csv(OUTPUT_CSV)
    
    article_counters, existing_ids = scan_existing_history(OUTPUT_CSV, OUTPUT_TXT_DIR)

    if existing_ids:
        print(f"CSV/TXT: Resumed — {len(existing_ids)} records synchronized [进度已同步].")
    else:
        print(f"TXT: Will create article text files in [将创建单篇正文文件] — {OUTPUT_TXT_DIR}")

    txt_files = sorted(glob.glob(os.path.join(TXT_DIR, "*.txt")))
    if not txt_files:
        print(f"  [Info / 提示] No .txt files found in {TXT_DIR} / 未找到txt文件")
        return
    print(f"\nFound {len(txt_files)} source files / 找到 {len(txt_files)} 个源文件.\n")

    processed_ts = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    outer_break  = False

    for txt_path in txt_files:
        if outer_break:
            break

        fname = os.path.basename(txt_path)
        print("\n" + "=" * 60)
        print(f"FILE / 文件: {fname}")
        print("=" * 60)

        with open(txt_path, encoding="utf-8", errors="replace") as f:
            full_text = f.read()

        file_meta  = parse_file_header(full_text, fname)
        page_index = build_page_index(full_text)
        all_seqs   = sorted(page_index.keys())

        if not all_seqs:
            print("  [Info / 提示] No page markers found. Skipping. / 未发现页码标记，跳过。")
            continue

        last_seq = all_seqs[-1]
        htid     = file_meta["htid"]
        volume   = file_meta["roman_numeral"] or file_meta["volume_label"] or ""
        source   = file_meta["source_file"]
        
        guessed_start_year, guessed_end_year = parse_year_range_from_filename(fname, file_meta["journal_year"])
        default_year_range = (
            f"{guessed_start_year}-{guessed_end_year}"
            if guessed_start_year and guessed_end_year and guessed_start_year != guessed_end_year
            else str(guessed_start_year or "")
        )
        while True:
            raw_year_range = input(
                f"  Confirm year range / 确认年份范围 (Enter for '{default_year_range}' / 回车使用该值): "
            ).strip()
            start_year, end_year = parse_year_range_input(raw_year_range, guessed_start_year, guessed_end_year)
            if start_year and end_year:
                break
            print("  [Error / 错误] Use YYYY or YYYY-YYYY, e.g. 1880-1881 / 请用YYYY或YYYY-YYYY格式。")

        year_label = f"{start_year}-{end_year}" if start_year != end_year else str(start_year)
        counter_key = f"{volume}_{year_label}"
        article_counters.setdefault(counter_key, 0)
        
        print(f"\n  HTID       : {htid}")
        print(f"  Volume     : {volume}")
        print(f"  Year range : {start_year}-{end_year}")
        print(f"  Total pages: {len(all_seqs)}  (seq {all_seqs[0]}-{last_seq})")
        print(f"  Base index / 当前起始序号: {article_counters[counter_key]} (existing records / 已有记录)")
        print()
        print("  Seq mapping (first 30) / seq页码映射（前30项）:")
        show_seq_table(page_index)
        print()
        print("  Commands / 命令: <seq> | done | quit | skip | show | show all | preview <seq>")

        while True:
            raw = input("\n  Start seq of next article / 下一篇文章起始seq (or done/quit/skip): ").strip()

            if raw.lower() == "quit":
                print("  Stopping. All files saved. / 正在停止，进度已保存。")
                outer_break = True
                break
            if raw.lower() == "done":
                print(f"  -> Done with {fname} / 当前文件处理完毕。")
                break
            if raw.lower() == "skip":
                print(f"  -> Skipping {fname} / 跳过当前文件。")
                break
            if raw.lower() == "show":
                show_seq_table(page_index)
                continue
            if raw.lower() == "show all":
                show_seq_table_full(page_index)
                continue
            if raw.lower().startswith("preview "):
                try:
                    pv_seq = int(raw.split()[1])
                    pv_pos = seq_to_char_pos(pv_seq, page_index)
                    if pv_pos is not None:
                        print(repr(full_text[pv_pos: pv_pos + 600]))
                    else:
                        print(f"  [Warning / 警告] seq {pv_seq} not found / 未找到该seq。")
                except Exception:
                    print("  Usage / 用法: preview <seq>")
                continue

            try:
                seq_start = int(raw)
            except ValueError:
                print("  [Error / 错误] Invalid input. Enter an integer or a command / 请输入数字或命令。")
                continue
            if seq_start not in page_index:
                print(f"  [Error / 错误] seq {seq_start} not found / 未找到该seq. Valid range / 有效范围: {all_seqs[0]}-{last_seq}")
                continue

            raw2 = input("  End seq / 结束seq (Enter for end-of-file / 回车表示到文件末尾): ").strip()
            defer_cmd = None

            if raw2.lower() in ("quit", "done", "skip"):
                seq_end   = None
                defer_cmd = raw2.lower()
            elif raw2 == "":
                seq_end = None
            else:
                try:
                    seq_end = int(raw2)
                except ValueError:
                    print("  [Error / 错误] Invalid end seq. Skipping this article / 结束seq无效，跳过此篇。")
                    continue
                if seq_end not in page_index:
                    print(f"  [Error / 错误] seq {seq_end} not found. Skipping this article / 未找到结束seq，跳过此篇。")
                    continue

            char_start   = page_index[seq_start]["char_pos"]
            char_end     = page_index[seq_end]["char_pos"] if seq_end is not None else len(full_text)
            article_text = full_text[char_start:char_end]
            print_start  = page_index[seq_start]["print_page"]

            seqs_in_article = (
                [s for s in all_seqs if seq_start <= s < seq_end]
                if seq_end is not None
                else [s for s in all_seqs if s >= seq_start]
            )
            actual_seq_end = seqs_in_article[-1] if seqs_in_article else seq_start
            print_end      = page_index[actual_seq_end]["print_page"]

            counter_key = f"{volume}_{year_label}"
            article_counters.setdefault(counter_key, 0)
            article_counters[counter_key] += 1
            idx        = article_counters[counter_key]
            article_id = f"{volume}_{year_label}_{idx:02d}"
            
            if article_id in existing_ids:
                print(f"  [Warning / 警告] {article_id} already exists. Stepping index... / ID已存在，自动递增。")
                while article_id in existing_ids:
                    article_counters[counter_key] += 1
                    idx = article_counters[counter_key]
                    article_id = f"{volume}_{year_label}_{idx:02d}"

            print(f"  Extracting metadata for [{article_id}] / 正在提取元数据...", end=" ", flush=True)
            meta = gemini_extract(article_text, api_key)
            time.sleep(RATE_LIMIT_SLEEP)
            print("Done / 完成。")

            page_url   = f"{HATHI_BASE}?id={htid}&seq={seq_start}" if htid and seq_start else ""

            article_txt_path = write_article_txt(OUTPUT_TXT_DIR, article_id, article_text)

            csv_row = {
                "article_id":       article_id,
                "source_file":      source,
                "htid":             htid,
                "start_year":       start_year,
                "end_year":         end_year,
                "volume_label":     file_meta["volume_label"],
                "roman_numeral":    file_meta["roman_numeral"],
                "title":            meta.get("title") or "",
                "author":           meta.get("author") or "",
                "author_title":     meta.get("author_title") or "",
                "location":         meta.get("location") or "",
                "contributor_role": meta.get("contributor_role") or "unknown",
                "section_type":     meta.get("section_type") or "article",
                "page_start":       print_start,
                "page_end":         print_end,
                "seq_start":        seq_start,
                "seq_end":          actual_seq_end,
                "page_url":         page_url,
                "fulltext_path":    article_txt_path,
                "char_count":       len(article_text),
                "parse_flag":       "OK" if meta else "PARSE_ERROR",
                "parse_notes":      "" if meta else "Gemini returned empty result",
                "processed_at":     processed_ts,
            }
            append_csv_row(OUTPUT_CSV, csv_row)
            existing_ids.add(article_id)

            print(f"  [OK] [{article_id}] {meta.get('title', '(no title)')[:60]}")
            print(f"       Author / 作者 : {meta.get('author', '-')}")
            print(f"       Pages  / 页码 : {print_start}-{print_end}  |  seq {seq_start}-{actual_seq_end}")
            print(f"       Saved  [状态] : CSV row + TXT file")

            if defer_cmd == "quit":
                outer_break = True
                break
            if defer_cmd in ("done", "skip"):
                break

    print("\n" + "=" * 60)
    print("All done / 全部处理完毕。")
    print(f"  CSV : {OUTPUT_CSV}")
    print(f"  TXT : {OUTPUT_TXT_DIR}")

if __name__ == "__main__":
    main()
