from __future__ import annotations

import json
import math
import sys
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd


ROOT = Path(r"C:\ras_text_analysis")
EXPANDED_CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "RAS_author_subject_year_1858-1948_utf8_simplified.csv"
OUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT
SUFFIX = sys.argv[3] if len(sys.argv) > 3 else ""

OUT_HTML = OUT_DIR / f"subject_time_evolution_subject_expanded{SUFFIX}.html"
BUNDLE_DIR = OUT_DIR / f"ras-subject-evolution_subject_expanded{SUFFIX}"
BUNDLE_HTML = BUNDLE_DIR / "index.html"

L1_TIME_CSV = OUT_DIR / f"subject_l1_time_subject_expanded{SUFFIX}.csv"
L2_TIME_CSV = OUT_DIR / f"subject_l2_time_subject_expanded{SUFFIX}.csv"
DIVERSITY_CSV = OUT_DIR / f"subject_diversity_subject_expanded{SUFFIX}.csv"
CENTRALITY_CSV = OUT_DIR / f"subject_network_centrality_time_subject_expanded{SUFFIX}.csv"
FULL_SUBJECT_DIST_CSV = OUT_DIR / f"subject_distribution_all_years_subject_expanded{SUFFIX}.csv"
SUBJECT_DIST_WINDOW_CSV = OUT_DIR / f"subject_distribution_time_window_subject_expanded{SUFFIX}.csv"
TITLE_TERM_TIME_CSV = OUT_DIR / f"title_term_time_subject_expanded{SUFFIX}.csv"
TITLE_BIGRAM_TIME_CSV = OUT_DIR / f"title_bigram_time_subject_expanded{SUFFIX}.csv"
TITLE_TFIDF_TIME_CSV = OUT_DIR / f"title_tfidf_time_subject_expanded{SUFFIX}.csv"
TITLE_TERM_SUBJECT_TIME_CSV = OUT_DIR / f"title_term_subject_time_subject_expanded{SUFFIX}.csv"
TITLE_PLACE_TIME_CSV = OUT_DIR / f"title_place_time_subject_expanded{SUFFIX}.csv"
TITLE_PLACE_SUBJECT_TIME_CSV = OUT_DIR / f"title_place_subject_time_subject_expanded{SUFFIX}.csv"
ARTICLE_TITLE_PLACES_CSV = OUT_DIR / f"article_title_places_subject_expanded{SUFFIX}.csv"
TITLE_PLACE_MAPPING_CSV = OUT_DIR / f"title_place_name_mapping_subject_expanded{SUFFIX}.csv"
CHINA_GEOJSON = ROOT / "china-geojson-master" / "china.json"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(EXPANDED_CSV, encoding="utf-8")
    df["year_start"] = pd.to_numeric(df["year_start"], errors="coerce")
    df["subject_weight"] = pd.to_numeric(df["subject_weight"], errors="coerce").fillna(0)
    df = df[
        (df["subject_l1"] != "Unknown")
        & (df["subject_l2"] != "Unknown")
        & df["subject_l1"].notna()
        & df["subject_l2"].notna()
    ].copy()
    df = df.drop_duplicates(subset=["master_id", "subject_l2"])
    return df


def load_all_subject_data() -> pd.DataFrame:
    df = pd.read_csv(EXPANDED_CSV, encoding="utf-8")
    df["year_start"] = pd.to_numeric(df["year_start"], errors="coerce")
    df["subject_weight"] = pd.to_numeric(df["subject_weight"], errors="coerce").fillna(0)
    df = df[
        (df["subject_l1"] != "Unknown")
        & (df["subject_l2"] != "Unknown")
        & df["subject_l1"].notna()
        & df["subject_l2"].notna()
    ].copy()
    return df.drop_duplicates(subset=["master_id", "subject_l2"])


def build_full_subject_distribution(df: pd.DataFrame) -> pd.DataFrame:
    dist = (
        df.groupby(["subject_l1", "subject_l2"], as_index=False)
        .agg(
            weighted_article_count=("subject_weight", "sum"),
            article_count=("master_id", "nunique"),
            author_count=("author_raw", "nunique"),
            first_year=("year_start", "min"),
            last_year=("year_start", "max"),
        )
        .sort_values(["weighted_article_count", "article_count", "subject_l2"], ascending=[False, False, True])
    )
    total = dist["weighted_article_count"].sum()
    dist["share"] = dist["weighted_article_count"] / total if total else 0
    return dist


def build_subject_distribution_by_window(df: pd.DataFrame) -> pd.DataFrame:
    dist = (
        df.groupby(["time_window", "subject_l1", "subject_l2"], as_index=False)
        .agg(
            weighted_article_count=("subject_weight", "sum"),
            article_count=("master_id", "nunique"),
            author_count=("author_raw", "nunique"),
            first_year=("year_start", "min"),
            last_year=("year_start", "max"),
        )
        .sort_values(["time_window", "weighted_article_count", "article_count", "subject_l2"], ascending=[True, False, False, True])
    )
    return add_shares(dist, ["time_window"])


def add_shares(summary: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    totals = summary.groupby(group_cols)["weighted_article_count"].sum().rename("window_total")
    summary = summary.merge(totals, on=group_cols, how="left")
    summary["share"] = summary["weighted_article_count"] / summary["window_total"]
    return summary.drop(columns=["window_total"])


def build_time_summaries(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    l1 = (
        df.groupby(["time_window", "subject_l1"], as_index=False)
        .agg(
            weighted_article_count=("subject_weight", "sum"),
            article_count=("master_id", "nunique"),
            subject_l2_count=("subject_l2", "nunique"),
            author_count=("author_raw", "nunique"),
        )
        .sort_values(["time_window", "subject_l1"])
    )
    l1 = add_shares(l1, ["time_window"])

    l2 = (
        df.groupby(["time_window", "subject_l1", "subject_l2"], as_index=False)
        .agg(
            weighted_article_count=("subject_weight", "sum"),
            article_count=("master_id", "nunique"),
            author_count=("author_raw", "nunique"),
        )
        .sort_values(["time_window", "subject_l1", "subject_l2"])
    )
    l2 = add_shares(l2, ["time_window"])

    diversity_rows = []
    for window, g in l2.groupby("time_window"):
        weights = g["weighted_article_count"].astype(float)
        total = weights.sum()
        shares = weights / total if total else weights
        shares = shares[shares > 0]
        shannon = float(-(shares * shares.map(math.log)).sum()) if len(shares) else 0.0
        hhi = float((shares**2).sum()) if len(shares) else 0.0
        top = sorted(shares.tolist(), reverse=True)
        diversity_rows.append(
            {
                "time_window": window,
                "weighted_article_count": total,
                "article_count": int(g["article_count"].sum()),
                "n_subject_l1": int(l1[l1["time_window"] == window]["subject_l1"].nunique()),
                "n_subject_l2": int(g["subject_l2"].nunique()),
                "shannon_entropy": shannon,
                "hhi": hhi,
                "top3_share": float(sum(top[:3])),
                "top5_share": float(sum(top[:5])),
            }
        )
    diversity = pd.DataFrame(diversity_rows).sort_values("time_window")
    return l1, l2, diversity


def subject_betweenness(edges: list[tuple[str, str]], subject_nodes: set[str]) -> dict[str, float]:
    graph: dict[str, set[str]] = defaultdict(set)
    for a, s in edges:
        graph[a].add(s)
        graph[s].add(a)

    nodes = list(graph.keys())
    cb = dict.fromkeys(nodes, 0.0)

    for source in nodes:
        stack = []
        pred = {w: [] for w in nodes}
        sigma = dict.fromkeys(nodes, 0.0)
        dist = dict.fromkeys(nodes, -1)
        sigma[source] = 1.0
        dist[source] = 0
        queue = deque([source])

        while queue:
            v = queue.popleft()
            stack.append(v)
            for w in graph[v]:
                if dist[w] < 0:
                    queue.append(w)
                    dist[w] = dist[v] + 1
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)

        delta = dict.fromkeys(nodes, 0.0)
        while stack:
            w = stack.pop()
            for v in pred[w]:
                if sigma[w]:
                    delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != source:
                cb[w] += delta[w]

    # Undirected graph normalization by 2.
    return {s: cb.get(s, 0.0) / 2 for s in subject_nodes}


def build_centrality(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for window, g in df.groupby("time_window"):
        pairs = g[["author_raw", "subject_l2"]].dropna().drop_duplicates()
        pairs = pairs[(pairs["author_raw"] != "") & (pairs["subject_l2"] != "")]
        subjects = set(pairs["subject_l2"])
        bet = subject_betweenness(list(pairs.itertuples(index=False, name=None)), subjects)

        agg = (
            g.groupby(["subject_l1", "subject_l2"], as_index=False)
            .agg(
                strength=("subject_weight", "sum"),
                article_count=("master_id", "nunique"),
                author_count=("author_raw", "nunique"),
            )
            .sort_values(["subject_l1", "subject_l2"])
        )
        for item in agg.to_dict("records"):
            rows.append(
                {
                    "time_window": window,
                    "subject_l1": item["subject_l1"],
                    "subject_l2": item["subject_l2"],
                    "strength": float(item["strength"]),
                    "article_count": int(item["article_count"]),
                    "author_count": int(item["author_count"]),
                    "degree": int(item["author_count"]),
                    "betweenness": float(bet.get(item["subject_l2"], 0.0)),
                }
            )
    return pd.DataFrame(rows).sort_values(["time_window", "subject_l1", "subject_l2"])


def build_articles(df: pd.DataFrame) -> pd.DataFrame:
    articles = df[
        [
            "time_window",
            "subject_l1",
            "subject_l2",
            "master_id",
            "author_raw",
            "title",
            "volume",
            "year_start",
        ]
    ].drop_duplicates()
    articles = articles.sort_values(["time_window", "subject_l1", "subject_l2", "year_start", "title"])
    return articles


def read_optional_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, encoding="utf-8")
    return pd.DataFrame()


def records(df: pd.DataFrame) -> list[dict]:
    clean = df.copy()
    clean = clean.where(pd.notna(clean), None)
    return clean.to_dict("records")


def read_geojson(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"type": "FeatureCollection", "features": []}


def build_html(
    l1: pd.DataFrame,
    l2: pd.DataFrame,
    full_subject_distribution: pd.DataFrame,
    subject_distribution_by_window: pd.DataFrame,
    diversity: pd.DataFrame,
    centrality: pd.DataFrame,
    articles: pd.DataFrame,
    all_articles: pd.DataFrame,
    title_terms: pd.DataFrame,
    title_bigrams: pd.DataFrame,
    title_tfidf: pd.DataFrame,
    title_subject_terms: pd.DataFrame,
    article_title_terms: pd.DataFrame,
    title_places: pd.DataFrame,
    title_subject_places: pd.DataFrame,
    article_title_places: pd.DataFrame,
    title_place_mapping: pd.DataFrame,
    year_span: str,
) -> str:
    payload = {
        "l1": records(l1),
        "l2": records(l2),
        "fullSubjectDistribution": records(full_subject_distribution),
        "subjectDistributionByWindow": records(subject_distribution_by_window),
        "diversity": records(diversity),
        "centrality": records(centrality),
        "articles": records(articles),
        "allArticles": records(all_articles),
        "titleTerms": records(title_terms),
        "titleBigrams": records(title_bigrams),
        "titleTfidf": records(title_tfidf),
        "titleSubjectTerms": records(title_subject_terms),
        "articleTitleTerms": records(article_title_terms),
        "titlePlaces": records(title_places),
        "titleSubjectPlaces": records(title_subject_places),
        "articleTitlePlaces": records(article_title_places),
        "titlePlaceMapping": records(title_place_mapping),
        "chinaGeojson": read_geojson(CHINA_GEOJSON),
    }
    data_json = json.dumps(payload, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RAS Subject Evolution, {year_span}</title>
<style>
:root {{
  --ink:#17202a; --muted:#64748b; --line:#d7dee8; --panel:#f7f9fc;
  --accent:#1f6f78; --bg:#ffffff;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:Arial, Helvetica, sans-serif; color:var(--ink); background:var(--bg); }}
.app {{ min-height:100vh; display:grid; grid-template-columns:minmax(0,1fr) 360px; }}
.main {{ min-width:0; border-right:1px solid var(--line); }}
.head {{ padding:16px 18px 12px; border-bottom:1px solid var(--line); }}
h1 {{ margin:0; font-size:20px; letter-spacing:0; }}
.sub {{ margin-top:5px; color:var(--muted); font-size:12px; }}
.tabs {{ display:flex; flex-wrap:wrap; gap:6px; margin-top:13px; }}
.tab {{ width:auto; height:34px; padding:0 12px; border:1px solid var(--line); border-radius:6px; background:#fff; color:#263241; cursor:pointer; font-weight:600; }}
.tab.active {{ background:#e7f2f3; border-color:#91b8bd; color:#174c54; }}
.filters {{ display:grid; grid-template-columns:1fr 1fr 1fr 1fr 1fr; gap:8px; margin-top:12px; }}
label {{ display:block; color:var(--muted); font-size:11px; margin-bottom:4px; }}
select, input {{ width:100%; height:34px; border:1px solid var(--line); border-radius:6px; padding:0 9px; background:#fff; color:var(--ink); }}
select:disabled, input:disabled {{ background:#eef2f6; color:#8a95a3; cursor:not-allowed; }}
.chart-wrap {{ padding:16px 18px 10px; }}
.chart-title {{ display:flex; justify-content:space-between; gap:12px; align-items:baseline; margin-bottom:8px; }}
.chart-title h2 {{ margin:0; font-size:16px; }}
.chart-note {{ color:var(--muted); font-size:12px; }}
svg {{ width:100%; height:560px; display:block; background:#fbfcfe; border:1px solid var(--line); border-radius:8px; }}
.legend {{ display:flex; flex-wrap:wrap; gap:8px 12px; padding:10px 0 0; }}
.chip {{ display:inline-flex; align-items:center; gap:5px; font-size:11px; color:#3c4858; }}
.dot {{ width:10px; height:10px; border-radius:50%; display:inline-block; }}
.side {{ display:flex; flex-direction:column; background:#fff; min-width:0; }}
.side-head {{ padding:16px; border-bottom:1px solid var(--line); }}
.side h2 {{ margin:0; font-size:17px; overflow-wrap:anywhere; }}
.side-sub {{ margin-top:5px; color:var(--muted); font-size:12px; line-height:1.35; }}
.stats {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; padding:12px 16px; background:var(--panel); border-bottom:1px solid var(--line); }}
.stat {{ background:#fff; border:1px solid var(--line); border-radius:6px; padding:8px; }}
.stat strong {{ display:block; font-size:17px; }}
.stat span {{ color:var(--muted); font-size:11px; }}
.list {{ overflow:auto; padding:12px 16px 18px; }}
.article {{ border-bottom:1px solid #edf1f6; padding:10px 0; }}
.article-title {{ font-size:13px; line-height:1.35; font-weight:600; }}
.article-meta {{ margin-top:5px; color:var(--muted); font-size:12px; line-height:1.35; }}
.tooltip {{ position:fixed; pointer-events:none; background:#13202f; color:#fff; border-radius:6px; padding:7px 9px; font-size:12px; max-width:300px; display:none; z-index:3; }}
@media (max-width:900px) {{
  .app {{ grid-template-columns:1fr; }}
  .main {{ border-right:0; }}
  .side {{ border-top:1px solid var(--line); min-height:420px; }}
  .filters {{ grid-template-columns:1fr; }}
  svg {{ height:520px; }}
}}
</style>
</head>
<body>
<div class="app">
  <main class="main">
    <div class="head">
      <h1>RAS Subject Evolution, {year_span}</h1>
      <div class="sub">Standardized subject trends across the full indexed journal range. Unknown subjects are excluded; weighted counts avoid double-counting multi-subject articles.</div>
      <div class="tabs">
        <button class="tab active" data-view="overview">Overview</button>
        <button class="tab" data-view="subjects">Subject Heatmap</button>
        <button class="tab" data-view="allSubjects">Subject Bar Chart</button>
        <button class="tab" data-view="diversity">Diversity</button>
        <button class="tab" data-view="network">Network Role</button>
        <button class="tab" data-view="placeNames">Place Name Signals</button>
        <button class="tab" data-view="placeMap">Place Map</button>
      </div>
      <div class="filters">
        <div><label for="subjectArea">Subject area</label><select id="subjectArea"></select></div>
        <div><label for="subjectSelect">Subject</label><select id="subjectSelect"></select></div>
        <div><label for="termSelect">Place name</label><select id="termSelect"></select></div>
        <div><label for="timeWindowSelect">Time window</label><select id="timeWindowSelect"></select></div>
        <div><label for="search">Article search</label><input id="search" type="search" placeholder="Title or author"></div>
      </div>
    </div>
    <div class="chart-wrap">
      <div class="chart-title">
        <h2 id="chartTitle">Overview</h2>
        <div id="chartNote" class="chart-note"></div>
      </div>
      <svg id="chart" role="img" aria-label="Subject evolution chart"></svg>
      <div id="legend" class="legend"></div>
    </div>
  </main>
  <aside class="side">
    <div class="side-head">
      <h2 id="panelTitle">Select a subject or time window</h2>
      <div id="panelSub" class="side-sub">Click a chart element to see article titles, years, and subjects.</div>
    </div>
    <div class="stats">
      <div class="stat"><strong id="statA">0</strong><span id="statALabel">weighted articles</span></div>
      <div class="stat"><strong id="statB">0</strong><span id="statBLabel">articles</span></div>
    </div>
    <div id="articleList" class="list"></div>
  </aside>
</div>
<div id="tooltip" class="tooltip"></div>
<script>
const DATA = {data_json};
const colors = {{
  "Anthropology and Social Customs":"#7b4f9d",
  "Arts and Archaeology":"#b85c38",
  "Economy, Trade, and Technology":"#26736b",
  "Education":"#677a1f",
  "Geography and Travel":"#2271a5",
  "History and Chronology":"#8f5a12",
  "Literature and Language":"#685bc7",
  "Military":"#6b7280",
  "Natural Sciences":"#2f855a",
  "Politics, Government, and Law":"#9f3a4a",
  "Religion and Philosophy":"#b23a71"
}};
let view = "overview";
const svg = document.getElementById("chart");
const legend = document.getElementById("legend");
const tooltip = document.getElementById("tooltip");
const areaEl = document.getElementById("subjectArea");
const subjectEl = document.getElementById("subjectSelect");
const termEl = document.getElementById("termSelect");
const timeWindowEl = document.getElementById("timeWindowSelect");
const searchEl = document.getElementById("search");
const panelTitle = document.getElementById("panelTitle");
const panelSub = document.getElementById("panelSub");
const articleList = document.getElementById("articleList");
const statA = document.getElementById("statA");
const statB = document.getElementById("statB");
const statALabel = document.getElementById("statALabel");
const statBLabel = document.getElementById("statBLabel");
const chartTitle = document.getElementById("chartTitle");
const chartNote = document.getElementById("chartNote");
function uniq(xs) {{ return Array.from(new Set(xs.filter(Boolean))).sort(); }}
function fmt(x, d=1) {{ return Number(x || 0).toLocaleString(undefined, {{maximumFractionDigits:d}}); }}
function pct(x) {{ return (Number(x || 0) * 100).toFixed(1) + "%"; }}
function esc(s) {{ return String(s ?? "").replace(/[&<>"]/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}}[c])); }}
function setOptions(sel, values, allLabel) {{
  const current = sel.value;
  sel.innerHTML = `<option value="">${{allLabel}}</option>` + values.map(v => `<option value="${{esc(v)}}">${{esc(v)}}</option>`).join("");
  if (values.includes(current)) sel.value = current;
}}
setOptions(areaEl, uniq(DATA.l1.map(d => d.subject_l1)), "All subject areas");
setOptions(timeWindowEl, uniq(DATA.l1.map(d => d.time_window)), "All windows");
function refreshSubjectOptions() {{
  const area = areaEl.value;
  const subjects = uniq(DATA.l2.filter(d => !area || d.subject_l1 === area).map(d => d.subject_l2));
  setOptions(subjectEl, subjects, "All subjects");
}}
refreshSubjectOptions();
function filteredPlaceNames() {{
  const area = areaEl.value, subj = subjectEl.value;
  const win = timeWindowEl.value;
  const source = (area || subj) ? DATA.titleSubjectPlaces.filter(d => (!area || d.subject_l1 === area) && (!subj || d.subject_l2 === subj)) : DATA.titlePlaces;
  return source.filter(d => !win || d.time_window === win);
}}
function refreshTermOptions() {{
  const terms = uniq(filteredPlaceNames().map(d => d.place_label));
  setOptions(termEl, terms, "Top place names");
}}
refreshTermOptions();
function svgSize() {{ const r = svg.getBoundingClientRect(); return {{w: Math.max(720, r.width), h: Math.max(420, r.height)}}; }}
function clear() {{ svg.innerHTML = ""; legend.innerHTML = ""; }}
function add(tag, attrs, parent=svg) {{
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k,v] of Object.entries(attrs || {{}})) el.setAttribute(k, v);
  parent.appendChild(el);
  return el;
}}
function showTip(ev, html) {{
  tooltip.innerHTML = html; tooltip.style.display = "block";
  tooltip.style.left = Math.min(window.innerWidth - 320, ev.clientX + 12) + "px";
  tooltip.style.top = (ev.clientY + 12) + "px";
}}
function hideTip() {{ tooltip.style.display = "none"; }}
function articlesFor(filter) {{
  const q = searchEl.value.trim().toLowerCase();
  const source = filter.all_years ? DATA.allArticles : DATA.articles;
  let allowedIds = null;
  if (filter.place_label) {{
    allowedIds = new Set(DATA.articleTitlePlaces.filter(t => t.place_label === filter.place_label).map(t => t.master_id));
  }}
  if (filter.place_labels) {{
    const labels = new Set(filter.place_labels);
    allowedIds = new Set(DATA.articleTitlePlaces.filter(t => labels.has(t.place_label)).map(t => t.master_id));
  }}
  return source.filter(a => {{
    if (filter.time_window && a.time_window !== filter.time_window) return false;
    if (filter.subject_l1 && a.subject_l1 !== filter.subject_l1) return false;
    if (filter.subject_l2 && a.subject_l2 !== filter.subject_l2) return false;
    if (allowedIds && !allowedIds.has(a.master_id)) return false;
    if (q && !(`${{a.title}} ${{a.author_raw}}`.toLowerCase().includes(q))) return false;
    return true;
  }}).sort((a,b) => Number(a.year_start)-Number(b.year_start) || String(a.title).localeCompare(String(b.title)));
}}
function updatePanel(title, subtitle, rows, a, b, aLabel="weighted articles", bLabel="articles") {{
  panelTitle.textContent = title;
  panelSub.textContent = subtitle;
  statA.textContent = fmt(a);
  statB.textContent = fmt(b, 0);
  statALabel.textContent = aLabel;
  statBLabel.textContent = bLabel;
  articleList.innerHTML = rows.slice(0, 160).map(r => `<div class="article"><div class="article-title">${{esc(r.title)}}</div><div class="article-meta">${{esc(r.volume || "")}}${{r.volume ? " - " : ""}}${{r.year_start}} - ${{esc(r.subject_l2)}} - ${{esc(r.subject_l1)}} - ${{esc(r.author_raw)}}</div></div>`).join("") || `<div class="article-meta">No matching articles.</div>`;
}}
function drawAxes(xLabels, yMax, margin, w, h, yFmt=fmt) {{
  add("line", {{x1:margin.l, y1:h-margin.b, x2:w-margin.r, y2:h-margin.b, stroke:"#8b98a8"}});
  add("line", {{x1:margin.l, y1:margin.t, x2:margin.l, y2:h-margin.b, stroke:"#8b98a8"}});
  for (let i=0; i<5; i++) {{
    const y = margin.t + (h-margin.t-margin.b) * i / 4;
    const val = yMax * (1 - i/4);
    add("line", {{x1:margin.l, y1:y, x2:w-margin.r, y2:y, stroke:"#e6ebf2"}});
    const t = add("text", {{x:margin.l-8, y:y+4, "text-anchor":"end", "font-size":11, fill:"#64748b"}});
    t.textContent = yFmt(val);
  }}
  xLabels.forEach((lab, i) => {{
    const x = margin.l + (w-margin.l-margin.r) * (i + 0.5) / xLabels.length;
    const t = add("text", {{x, y:h-margin.b+18, "text-anchor":"middle", "font-size":10, fill:"#64748b"}});
    t.textContent = lab;
  }});
}}
function drawLegend(items) {{
  legend.innerHTML = items.map(([label,color]) => `<span class="chip"><span class="dot" style="background:${{color}}"></span>${{esc(label)}}</span>`).join("");
}}
function filteredL1() {{
  const area = areaEl.value;
  return DATA.l1.filter(d => !area || d.subject_l1 === area);
}}
function filteredL2() {{
  const area = areaEl.value, subj = subjectEl.value;
  return DATA.l2.filter(d => (!area || d.subject_l1 === area) && (!subj || d.subject_l2 === subj));
}}
function drawOverview() {{
  clear(); chartTitle.textContent = "Overview: subject-area share over time"; chartNote.textContent = "Stacked share by first-level subject";
  const {{w,h}} = svgSize(); svg.setAttribute("viewBox", `0 0 ${{w}} ${{h}}`);
  const margin = {{l:68,r:24,t:24,b:58}};
  const windows = uniq(DATA.l1.map(d => d.time_window));
  const areas = uniq(filteredL1().map(d => d.subject_l1));
  drawAxes(windows, 1, margin, w, h, pct);
  const xStep = (w-margin.l-margin.r)/windows.length;
  windows.forEach((win, i) => {{
    let yBase = h - margin.b;
    areas.forEach(area => {{
      const d = DATA.l1.find(r => r.time_window === win && r.subject_l1 === area);
      if (!d) return;
      const barH = Number(d.share) * (h-margin.t-margin.b);
      const rect = add("rect", {{x:margin.l+i*xStep+5, y:yBase-barH, width:Math.max(1,xStep-10), height:barH, fill:colors[area] || "#777", opacity:.9}});
      rect.addEventListener("mousemove", ev => showTip(ev, `<b>${{esc(area)}}</b><br>${{win}}<br>Share: ${{pct(d.share)}}<br>Weighted articles: ${{fmt(d.weighted_article_count)}}`));
      rect.addEventListener("mouseleave", hideTip);
      rect.addEventListener("click", () => updatePanel(area, `${{win}} - ${{pct(d.share)}} of weighted subject activity`, articlesFor({{time_window:win, subject_l1:area}}), d.weighted_article_count, d.article_count));
      yBase -= barH;
    }});
  }});
  drawLegend(areas.map(a => [a, colors[a] || "#777"]));
}}
function drawSubjects() {{
  clear(); chartTitle.textContent = "Subject Heatmap: weighted article count"; chartNote.textContent = "Each cell shows a subject's weighted article count within a time window; top subjects are shown unless a subject is selected";
  const {{w,h}} = svgSize(); svg.setAttribute("viewBox", `0 0 ${{w}} ${{h}}`);
  const margin = {{l:178,r:32,t:20,b:58}};
  let data = filteredL2();
  const selected = subjectEl.value;
  let subjects = selected ? [selected] : uniq(data.map(d => d.subject_l2)).sort((a,b) => {{
    const sa = data.filter(d => d.subject_l2===a).reduce((s,d)=>s+Number(d.weighted_article_count),0);
    const sb = data.filter(d => d.subject_l2===b).reduce((s,d)=>s+Number(d.weighted_article_count),0);
    return sb-sa;
  }}).slice(0, 24);
  data = data.filter(d => subjects.includes(d.subject_l2));
  const windows = uniq(DATA.l2.map(d => d.time_window));
  const maxVal = Math.max(1, ...data.map(d => Number(d.weighted_article_count)));
  const cellW = (w-margin.l-margin.r)/windows.length;
  const cellH = Math.max(12, (h-margin.t-margin.b)/subjects.length);
  windows.forEach((win,i) => {{
    const t = add("text", {{x:margin.l+i*cellW+cellW/2, y:h-margin.b+18, "text-anchor":"middle", "font-size":10, fill:"#64748b"}});
    t.textContent = win;
  }});
  subjects.forEach((subj,j) => {{
    const label = add("text", {{x:margin.l-8, y:margin.t+j*cellH+cellH*.65, "text-anchor":"end", "font-size":11, fill:"#263241"}});
    label.textContent = subj.length > 24 ? subj.slice(0,23)+"..." : subj;
    windows.forEach((win,i) => {{
      const d = data.find(r => r.time_window===win && r.subject_l2===subj);
      const val = d ? Number(d.weighted_article_count) : 0;
      const op = val ? 0.18 + 0.82 * Math.sqrt(val/maxVal) : 0.04;
      const area = d?.subject_l1 || DATA.l2.find(r => r.subject_l2===subj)?.subject_l1;
      const rect = add("rect", {{x:margin.l+i*cellW+1, y:margin.t+j*cellH+1, width:Math.max(1,cellW-2), height:Math.max(1,cellH-2), fill:colors[area] || "#777", opacity:op}});
      rect.addEventListener("mousemove", ev => showTip(ev, `<b>${{esc(subj)}}</b><br>${{win}}<br>Weighted articles: ${{fmt(val)}}<br>Share: ${{d ? pct(d.share) : "0.0%"}}`));
      rect.addEventListener("mouseleave", hideTip);
      rect.addEventListener("click", () => updatePanel(subj, `${{win}} - ${{esc(area || "")}}`, articlesFor({{time_window:win, subject_l2:subj}}), val, d ? d.article_count : 0));
    }});
  }});
  drawLegend(uniq(data.map(d => d.subject_l1)).map(a => [a, colors[a] || "#777"]));
}}
function drawAllSubjects() {{
  clear();
  const win = timeWindowEl.value;
  chartTitle.textContent = `${{win || "All-Time"}} Subject Bar Chart${{win ? "" : ", {year_span}"}}`;
  chartNote.textContent = "Static distribution for the selected time window; bar length uses weighted article count.";
  const {{w,h}} = svgSize(); svg.setAttribute("viewBox", `0 0 ${{w}} ${{h}}`);
  const area = areaEl.value;
  let rows = (win ? DATA.subjectDistributionByWindow.filter(d => d.time_window === win) : DATA.fullSubjectDistribution).filter(d => !area || d.subject_l1 === area);
  rows = rows.slice().sort((a,b) => Number(b.weighted_article_count)-Number(a.weighted_article_count) || String(a.subject_l2).localeCompare(String(b.subject_l2)));
  const shown = rows.slice(0, 42);
  const margin = {{l:220,r:70,t:20,b:36}};
  const barH = Math.max(10, Math.min(18, (h-margin.t-margin.b)/Math.max(1, shown.length) - 2));
  const gap = 3;
  const maxVal = Math.max(1, ...shown.map(d => Number(d.weighted_article_count || 0)));
  shown.forEach((d,i) => {{
    const y0 = margin.t + i*(barH+gap);
    const bw = Number(d.weighted_article_count || 0)/maxVal*(w-margin.l-margin.r);
    const label = add("text", {{x:margin.l-8, y:y0+barH*.72, "text-anchor":"end", "font-size":11, fill:"#263241"}});
    label.textContent = d.subject_l2.length > 30 ? d.subject_l2.slice(0,29)+"..." : d.subject_l2;
    const rect = add("rect", {{x:margin.l, y:y0, width:Math.max(1,bw), height:barH, fill:colors[d.subject_l1] || "#777", opacity:.82}});
    const val = add("text", {{x:margin.l+bw+5, y:y0+barH*.72, "font-size":10, fill:"#64748b"}});
    val.textContent = `${{fmt(d.weighted_article_count,1)}} / ${{d.article_count}}`;
    rect.addEventListener("mousemove", ev => showTip(ev, `<b>${{esc(d.subject_l2)}}</b><br>${{esc(d.subject_l1)}}<br>Weighted articles: ${{fmt(d.weighted_article_count,2)}}<br>Articles: ${{d.article_count}}<br>Years: ${{d.first_year}}-${{d.last_year}}<br>Authors: ${{d.author_count}}<br>Share: ${{pct(d.share)}}`));
    rect.addEventListener("mouseleave", hideTip);
    rect.addEventListener("click", () => updatePanel(d.subject_l2, `${{win || "All windows"}} - ${{d.first_year}}-${{d.last_year}} - ${{pct(d.share)}} of weighted subject activity`, articlesFor({{time_window:win, subject_l2:d.subject_l2, all_years:true}}), d.weighted_article_count, d.article_count));
  }});
  panelTitle.textContent = `${{win || "All-Time"}} Subject Bar Chart`;
  panelSub.textContent = `${{rows.length}} subjects. Showing top ${{shown.length}} bars; detailed distribution is also exported as CSV.`;
  statA.textContent = fmt(rows.reduce((s,d)=>s+Number(d.weighted_article_count || 0),0),1);
  statB.textContent = rows.length;
  statALabel.textContent = "weighted articles";
  statBLabel.textContent = "subjects";
  articleList.innerHTML = rows.map(d => `<div class="article"><div class="article-title">${{esc(d.subject_l2)}}</div><div class="article-meta">${{esc(d.subject_l1)}} - weighted ${{fmt(d.weighted_article_count,2)}} - articles ${{d.article_count}} - years ${{d.first_year}}-${{d.last_year}} - share ${{pct(d.share)}}</div></div>`).join("");
  drawLegend(uniq(rows.map(d => d.subject_l1)).map(a => [a, colors[a] || "#777"]));
}}
function drawDiversity() {{
  clear(); chartTitle.textContent = "Diversity: dispersion and concentration"; chartNote.textContent = "Calculated for all subjects in each time window; subject filters are disabled here.";
  const {{w,h}} = svgSize(); svg.setAttribute("viewBox", `0 0 ${{w}} ${{h}}`);
  const margin = {{l:70,r:34,t:24,b:58}};
  const windows = DATA.diversity.map(d => d.time_window);
  const series = [
    ["shannon_entropy", "#1f6f78", "Shannon entropy"],
    ["hhi", "#9f3a4a", "HHI"],
    ["top5_share", "#685bc7", "Top 5 share"]
  ];
  const yMax = Math.max(...DATA.diversity.flatMap(d => series.map(s => Number(d[s[0]]))));
  drawAxes(windows, yMax, margin, w, h, fmt);
  const x = i => margin.l + (w-margin.l-margin.r) * (i+.5)/windows.length;
  const y = val => h-margin.b - Number(val)/yMax*(h-margin.t-margin.b);
  series.forEach(([key,color,label]) => {{
    const pts = DATA.diversity.map((d,i) => `${{x(i)}},${{y(d[key])}}`).join(" ");
    add("polyline", {{points:pts, fill:"none", stroke:color, "stroke-width":2.5}});
    DATA.diversity.forEach((d,i) => {{
      const c = add("circle", {{cx:x(i), cy:y(d[key]), r:5, fill:color}});
      c.addEventListener("mousemove", ev => showTip(ev, `<b>${{label}}</b><br>${{d.time_window}}<br>${{fmt(d[key],3)}}<br>L2 subjects: ${{d.n_subject_l2}}`));
      c.addEventListener("mouseleave", hideTip);
      c.addEventListener("click", () => updatePanel(d.time_window, `Subject diversity: ${{fmt(d.shannon_entropy,2)}} Shannon; ${{fmt(d.hhi,3)}} HHI`, articlesFor({{time_window:d.time_window}}), d.weighted_article_count, d.n_subject_l2, "weighted articles", "subjects"));
    }});
  }});
  drawLegend(series.map(s => [s[2], s[1]]));
}}
function drawNetwork() {{
  clear(); chartTitle.textContent = "Network Role: subject strength and author reach"; chartNote.textContent = "Bubble size = connected authors; vertical position = network strength";
  const {{w,h}} = svgSize(); svg.setAttribute("viewBox", `0 0 ${{w}} ${{h}}`);
  const margin = {{l:70,r:34,t:24,b:58}};
  let data = DATA.centrality.filter(d => (!areaEl.value || d.subject_l1===areaEl.value) && (!subjectEl.value || d.subject_l2===subjectEl.value));
  if (!subjectEl.value) data = data.sort((a,b) => Number(b.strength)-Number(a.strength)).slice(0, 220);
  const windows = uniq(DATA.centrality.map(d => d.time_window));
  const yMax = Math.max(1, ...data.map(d => Number(d.strength)));
  drawAxes(windows, yMax, margin, w, h, fmt);
  const x = win => margin.l + (w-margin.l-margin.r) * (windows.indexOf(win)+.5)/windows.length;
  const y = val => h-margin.b - Number(val)/yMax*(h-margin.t-margin.b);
  data.forEach(d => {{
    const r = 3 + Math.sqrt(Number(d.author_count || 1))*2.2;
    const c = add("circle", {{cx:x(d.time_window), cy:y(d.strength), r, fill:colors[d.subject_l1] || "#777", opacity:.75, stroke:"#fff", "stroke-width":1}});
    c.addEventListener("mousemove", ev => showTip(ev, `<b>${{esc(d.subject_l2)}}</b><br>${{d.time_window}}<br>Strength: ${{fmt(d.strength)}}<br>Authors: ${{d.author_count}}<br>Betweenness: ${{fmt(d.betweenness,1)}}`));
    c.addEventListener("mouseleave", hideTip);
    c.addEventListener("click", () => updatePanel(d.subject_l2, `${{d.time_window}} - authors: ${{d.author_count}}, strength: ${{fmt(d.strength)}}`, articlesFor({{time_window:d.time_window, subject_l2:d.subject_l2}}), d.strength, d.author_count, "network strength", "authors"));
  }});
  drawLegend(uniq(data.map(d => d.subject_l1)).map(a => [a, colors[a] || "#777"]));
}}
function drawPlaceNames() {{
  clear(); chartTitle.textContent = "Place Name Signals: title place names over time"; chartNote.textContent = "Based on place names found in article titles and normalized through note-column annotations.";
  const {{w,h}} = svgSize(); svg.setAttribute("viewBox", `0 0 ${{w}} ${{h}}`);
  const margin = {{l:178,r:32,t:20,b:58}};
  let data = filteredPlaceNames();
  const selected = termEl.value;
  let places = selected ? [selected] : uniq(data.map(d => d.place_label)).sort((a,b) => {{
    const sa = data.filter(d => d.place_label===a).reduce((s,d)=>s+Number(d.tfidf || d.place_count),0);
    const sb = data.filter(d => d.place_label===b).reduce((s,d)=>s+Number(d.tfidf || d.place_count),0);
    return sb-sa;
  }}).slice(0, 26);
  data = data.filter(d => places.includes(d.place_label));
  const windows = uniq(DATA.titlePlaces.map(d => d.time_window));
  const maxVal = Math.max(1, ...data.map(d => Number(d.tfidf || d.place_count)));
  const cellW = (w-margin.l-margin.r)/windows.length;
  const cellH = Math.max(12, (h-margin.t-margin.b)/Math.max(1, places.length));
  windows.forEach((win,i) => {{
    const t = add("text", {{x:margin.l+i*cellW+cellW/2, y:h-margin.b+18, "text-anchor":"middle", "font-size":10, fill:"#64748b"}});
    t.textContent = win;
  }});
  places.forEach((place,j) => {{
    const label = add("text", {{x:margin.l-8, y:margin.t+j*cellH+cellH*.65, "text-anchor":"end", "font-size":11, fill:"#263241"}});
    label.textContent = place.length > 24 ? place.slice(0,23)+"..." : place;
    windows.forEach((win,i) => {{
      const candidates = data.filter(r => r.time_window===win && r.place_label===place);
      const d = candidates.length ? candidates.sort((a,b)=>Number(b.tfidf || b.place_count)-Number(a.tfidf || a.place_count))[0] : null;
      const val = d ? Number(d.tfidf || d.place_count) : 0;
      const op = val ? 0.16 + 0.84 * Math.sqrt(val/maxVal) : 0.04;
      const area = d?.subject_l1 || areaEl.value || "Place names";
      const fill = d?.subject_l1 ? (colors[d.subject_l1] || "#777") : "#1f6f78";
      const rect = add("rect", {{x:margin.l+i*cellW+1, y:margin.t+j*cellH+1, width:Math.max(1,cellW-2), height:Math.max(1,cellH-2), fill, opacity:op}});
      rect.addEventListener("mousemove", ev => showTip(ev, `<b>${{esc(place)}}</b><br>${{win}}<br>Score: ${{fmt(val,2)}}<br>Articles: ${{d ? d.article_count : 0}}`));
      rect.addEventListener("mouseleave", hideTip);
      rect.addEventListener("click", () => updatePanel(place, `${{win}} - place-name signal${{d?.subject_l2 ? " - " + d.subject_l2 : ""}}`, articlesFor({{time_window:win, subject_l1:d?.subject_l1 || areaEl.value, subject_l2:d?.subject_l2 || subjectEl.value, place_label:place}}), val, d ? d.article_count : 0, "score", "articles"));
    }});
  }});
  const topPlaces = DATA.titlePlaces.filter(d => !selected && (!areaEl.value && !subjectEl.value)).sort((a,b)=>Number(b.place_count)-Number(a.place_count)).slice(0, 12);
  const legendItems = topPlaces.length ? topPlaces.map(p => [`${{p.time_window}}: ${{p.place_label}}`, "#1f6f78"]) : [["Place names are filtered by subject controls when selected", "#1f6f78"]];
  drawLegend(legendItems);
}}
function drawPlaceMap() {{
  clear(); chartTitle.textContent = "Place Map: title place names by time"; chartNote.textContent = "Grey provinces mark where title-place mentions cluster; dots retain the normalized place-name detail.";
  const {{w,h}} = svgSize(); svg.setAttribute("viewBox", `0 0 ${{w}} ${{h}}`);
  const margin = {{l:30,r:34,t:18,b:30}};
  const area = areaEl.value, subj = subjectEl.value, selectedPlace = termEl.value, win = timeWindowEl.value;
  let data = (area || subj) ? DATA.titleSubjectPlaces.filter(d => (!area || d.subject_l1===area) && (!subj || d.subject_l2===subj)) : DATA.titlePlaces;
  data = data.filter(d => (!win || d.time_window===win) && (!selectedPlace || d.place_label===selectedPlace) && d.lat != null && d.lon != null);
  const placeProvince = name => {{
    const n = String(name || "").toLowerCase();
    if (/(guangdong|guangzhou|canton|lampacau|macau)/.test(n)) return "广东省";
    if (/(shandong|tai shan)/.test(n)) return "山东省";
    if (/(sichuan|chengdu|songpan|batang)/.test(n)) return "四川省";
    if (/(fujian|fuzhou|foochow)/.test(n)) return "福建省";
    if (/(gansu|kansu)/.test(n)) return "甘肃省";
    if (/(guangxi|kwangsi)/.test(n)) return "广西壮族自治区";
    if (/(guizhou|kweichow)/.test(n)) return "贵州省";
    if (/(hankou|yangtzi|hubei)/.test(n)) return "湖北省";
    if (/(henan|kaifeng|honan)/.test(n)) return "河南省";
    if (/(hunan)/.test(n)) return "湖南省";
    if (/(shaanxi|shen-hsi)/.test(n)) return "陕西省";
    if (/(shanxi|shan-hsi)/.test(n)) return "山西省";
    if (/(shaoxing|zhejiang)/.test(n)) return "浙江省";
    if (/(wusong|shanghai)/.test(n)) return "上海市";
    if (/(zhenjiang|jiangsu)/.test(n)) return "江苏省";
    if (/(taiwan|formosa)/.test(n)) return "台湾省";
    if (/(mangkang|menkong|tibet)/.test(n)) return "西藏自治区";
    if (/(pu'er|yunnan|szemao)/.test(n)) return "云南省";
    if (/(beijing|pekin)/.test(n)) return "北京市";
    if (/(sakhalin|tonkin|yangon|rangoon)/.test(n)) return null;
    return null;
  }};
  const grouped = new Map();
  data.forEach(d => {{
    const key = `${{d.modern_name}}|${{d.place_label}}|${{d.lat}}|${{d.lon}}`;
    const old = grouped.get(key) || {{modern_name:d.modern_name, place_label:d.place_label, province:placeProvince(d.modern_name || d.place_label), lat:Number(d.lat), lon:Number(d.lon), place_count:0, article_count:0, windows:new Set(), subject_l1:d.subject_l1, subject_l2:d.subject_l2}};
    old.place_count += Number(d.place_count || 0);
    old.article_count += Number(d.article_count || 0);
    old.windows.add(d.time_window);
    grouped.set(key, old);
  }});
  const places = Array.from(grouped.values());
  const lonMin = 73, lonMax = 145, latMin = 10, latMax = 55;
  const cosRef = Math.cos(35 * Math.PI / 180);
  const projX = lon => Number(lon) * cosRef;
  const pxMin = projX(lonMin), pxMax = projX(lonMax);
  const plotW = w - margin.l - margin.r, plotH = h - margin.t - margin.b;
  const mapScale = Math.min(plotW / (pxMax - pxMin), plotH / (latMax - latMin));
  const mapW = (pxMax - pxMin) * mapScale, mapH = (latMax - latMin) * mapScale;
  const mapLeft = margin.l + (plotW - mapW) / 2;
  const mapTop = margin.t + (plotH - mapH) / 2;
  const x = lon => mapLeft + (projX(lon)-pxMin) * mapScale;
  const y = lat => mapTop + (latMax-Number(lat)) * mapScale;
  add("rect", {{x:0, y:0, width:w, height:h, fill:"#ffffff"}});
  const geoFeatures = DATA.chinaGeojson?.features || [];
  if (geoFeatures.length) {{
    const provinceCounts = new Map();
    places.forEach(p => {{
      if (!p.province) return;
      const old = provinceCounts.get(p.province) || {{count:0, articles:0, labels:new Set()}};
      old.count += p.place_count;
      old.articles += p.article_count;
      old.labels.add(p.place_label);
      provinceCounts.set(p.province, old);
    }});
    const maxProvince = Math.max(1, ...Array.from(provinceCounts.values()).map(v => v.count));
    const ringPath = ring => ring.map((coord, i) => `${{i ? "L" : "M"}}${{x(coord[0])}},${{y(coord[1])}}`).join("") + "Z";
    const geometryPath = geom => {{
      if (!geom) return "";
      if (geom.type === "Polygon") return geom.coordinates.map(ringPath).join("");
      if (geom.type === "MultiPolygon") return geom.coordinates.map(poly => poly.map(ringPath).join("")).join("");
      return "";
    }};
    const mapGroup = add("g", {{}});
    geoFeatures.forEach(feature => {{
      const name = feature.properties?.name || "";
      const hit = provinceCounts.get(name);
      const tone = hit ? Math.round(238 - 82 * Math.sqrt(hit.count / maxProvince)) : 252;
      const fill = hit ? `rgb(${{tone}},${{tone}},${{tone}})` : "#fbfbfb";
      const path = add("path", {{d:geometryPath(feature.geometry), fill, stroke:"#b8bdc3", "stroke-width":1.05, "vector-effect":"non-scaling-stroke", "fill-rule":"evenodd"}}, mapGroup);
      if (hit) {{
        path.addEventListener("mousemove", ev => showTip(ev, `<b>${{esc(name)}}</b><br>${{hit.count}} title-place mentions<br>${{hit.articles}} article links<br>${{Array.from(hit.labels).slice(0,6).map(esc).join("<br>")}}`));
        path.addEventListener("mouseleave", hideTip);
        path.addEventListener("click", () => updatePanel(name, `${{win || "All windows"}} - province place signal`, articlesFor({{time_window:win, subject_l1:area, subject_l2:subj, place_labels:Array.from(hit.labels)}}), hit.count, hit.articles, "mentions", "articles"));
      }}
    }});
  }} else {{
  const provinceShapes = [
    ["Xinjiang", [[73.5,39.3],[75.2,48.2],[80.5,49.7],[88.8,49.1],[95.3,44.8],[94.0,39.6],[90.8,36.8],[84.2,36.2],[79.0,34.2],[75.0,36.0]]],
    ["Tibet", [[78.0,31.4],[82.5,35.8],[91.0,36.7],[97.5,34.0],[101.5,31.2],[98.8,28.2],[92.2,27.3],[86.3,28.5],[80.3,30.0]]],
    ["Qinghai", [[90.8,36.8],[94.0,39.6],[100.7,39.4],[104.2,36.1],[101.5,31.2],[97.5,34.0]]],
    ["Gansu", [[94.0,39.6],[100.7,39.4],[104.9,38.0],[106.6,35.3],[103.6,33.3],[101.5,31.2],[104.2,36.1]]],
    ["Inner Mongolia", [[97.0,42.4],[103.2,41.5],[109.8,41.3],[116.2,43.8],[122.4,48.5],[119.0,49.8],[110.5,47.2],[101.5,45.2]]],
    ["Heilongjiang", [[122.4,48.5],[126.0,53.5],[132.0,51.3],[134.6,47.5],[129.2,45.4]]],
    ["Jilin", [[124.4,42.4],[129.2,45.4],[134.6,47.5],[130.4,42.8],[126.0,41.2]]],
    ["Liaoning", [[119.5,40.0],[124.4,42.4],[126.0,41.2],[124.1,39.2],[121.2,38.5]]],
    ["Beijing", [[115.6,40.4],[116.8,40.7],[117.3,39.8],[116.4,39.2],[115.5,39.6]]],
    ["Tianjin", [[117.0,39.5],[118.1,39.4],[118.0,38.7],[117.0,38.8]]],
    ["Hebei", [[113.7,42.0],[116.2,43.8],[119.5,40.0],[121.2,38.5],[118.2,36.8],[114.8,37.0],[113.1,39.0]]],
    ["Shanxi", [[110.4,40.8],[113.1,39.0],[114.8,37.0],[112.4,34.8],[109.6,35.2],[108.8,38.4]]],
    ["Shandong", [[114.8,37.0],[118.2,36.8],[122.5,37.8],[121.2,35.7],[117.2,34.7],[114.2,35.2]]],
    ["Ningxia", [[105.1,38.8],[107.6,38.5],[107.2,36.1],[105.4,35.3],[104.9,38.0]]],
    ["Shaanxi", [[106.6,35.3],[109.6,35.2],[112.4,34.8],[111.5,31.8],[108.2,31.4],[105.6,33.3]]],
    ["Henan", [[112.4,34.8],[114.2,35.2],[117.2,34.7],[116.5,32.6],[113.8,31.4],[111.5,31.8]]],
    ["Jiangsu", [[117.2,34.7],[121.2,35.7],[121.8,32.0],[119.5,31.0],[116.5,32.6]]],
    ["Shanghai", [[120.9,31.5],[121.8,31.6],[121.8,30.9],[120.9,30.9]]],
    ["Anhui", [[116.5,32.6],[119.5,31.0],[119.0,29.2],[116.0,29.6],[113.8,31.4]]],
    ["Hubei", [[108.2,31.4],[111.5,31.8],[113.8,31.4],[116.0,29.6],[113.0,28.4],[109.5,29.2]]],
    ["Sichuan", [[101.5,31.2],[103.6,33.3],[105.6,33.3],[108.2,31.4],[107.4,28.0],[103.7,26.2],[99.0,28.1]]],
    ["Chongqing", [[106.0,31.2],[108.2,31.4],[107.4,28.0],[105.2,28.5]]],
    ["Hunan", [[109.5,29.2],[113.0,28.4],[114.0,25.8],[111.0,24.8],[108.5,26.0]]],
    ["Jiangxi", [[113.0,28.4],[116.0,29.6],[118.6,27.5],[117.0,24.8],[114.0,25.8]]],
    ["Zhejiang", [[119.0,29.2],[121.8,30.9],[122.0,28.0],[120.1,27.0],[118.6,27.5]]],
    ["Fujian", [[117.0,24.8],[118.6,27.5],[120.1,27.0],[119.8,24.0],[117.4,23.6]]],
    ["Guizhou", [[104.0,27.4],[108.5,26.0],[111.0,24.8],[109.0,22.8],[105.2,24.2]]],
    ["Yunnan", [[98.8,28.2],[103.7,26.2],[105.2,24.2],[103.2,21.6],[99.2,22.5],[97.5,24.6]]],
    ["Guangxi", [[105.2,24.2],[109.0,22.8],[112.2,22.2],[110.2,20.8],[106.5,21.8],[103.2,21.6]]],
    ["Guangdong", [[109.0,22.8],[111.0,24.8],[114.0,25.8],[117.4,23.6],[116.0,22.0],[112.2,22.2]]],
    ["Hainan", [[108.4,19.6],[110.2,20.4],[111.7,19.2],[110.6,18.2],[108.8,18.4]]],
    ["Taiwan", [[120.0,25.3],[121.5,25.2],[122.2,23.5],[121.2,21.9],[120.1,22.8]]]
  ];
  const provinceCounts = new Map();
  places.forEach(p => {{
    if (!p.province) return;
    const old = provinceCounts.get(p.province) || {{count:0, articles:0, labels:new Set()}};
    old.count += p.place_count;
    old.articles += p.article_count;
    old.labels.add(p.place_label);
    provinceCounts.set(p.province, old);
  }});
  const maxProvince = Math.max(1, ...Array.from(provinceCounts.values()).map(v => v.count));
  const mapGroup = add("g", {{}});
  provinceShapes.forEach(([name, pts]) => {{
    const hit = provinceCounts.get(name);
    const tone = hit ? Math.round(238 - 82 * Math.sqrt(hit.count / maxProvince)) : 252;
    const fill = hit ? `rgb(${{tone}},${{tone}},${{tone}})` : "#fbfbfb";
    const poly = add("polygon", {{points:pts.map(p => `${{x(p[0])}},${{y(p[1])}}`).join(" "), fill, stroke:"#b8bdc3", "stroke-width":1.1, "vector-effect":"non-scaling-stroke"}}, mapGroup);
    if (hit) {{
      poly.addEventListener("mousemove", ev => showTip(ev, `<b>${{esc(name)}}</b><br>${{hit.count}} title-place mentions<br>${{hit.articles}} article links<br>${{Array.from(hit.labels).slice(0,6).map(esc).join("<br>")}}`));
      poly.addEventListener("mouseleave", hideTip);
      poly.addEventListener("click", () => updatePanel(name, `${{win || "All windows"}} - province place signal`, articlesFor({{time_window:win, subject_l1:area, subject_l2:subj, place_labels:Array.from(hit.labels)}}), hit.count, hit.articles, "mentions", "articles"));
    }}
  }});
  }}
  const insetX = w - 160, insetY = h - 145;
  add("rect", {{x:insetX, y:insetY, width:74, height:92, fill:"#fff", stroke:"#aeb4bb", "stroke-width":1.4}});
  [[12,16,18,9],[47,12,55,3],[55,26,62,16],[18,48,22,38],[35,67,43,58],[51,78,58,68]].forEach(s => {{
    add("line", {{x1:insetX+s[0], y1:insetY+s[1], x2:insetX+s[2], y2:insetY+s[3], stroke:"#9da3aa", "stroke-width":1.8, "stroke-dasharray":"5 6", "stroke-linecap":"round"}});
  }});
  const maxCount = Math.max(1, ...places.map(p => p.place_count));
  places.forEach(p => {{
    if (!p.province) return;
    const r = 2.8 + Math.sqrt(p.place_count / maxCount) * 7.5;
    const c = add("circle", {{cx:x(p.lon), cy:y(p.lat), r, fill:"#2f3a46", opacity:.42, stroke:"#fff", "stroke-width":1.1}});
    c.addEventListener("mousemove", ev => showTip(ev, `<b>${{esc(p.place_label)}}</b><br>${{p.place_count}} mentions<br>${{p.article_count}} articles<br>${{Array.from(p.windows).sort().join(", ")}}`));
    c.addEventListener("mouseleave", hideTip);
    c.addEventListener("click", () => updatePanel(p.place_label, `${{win || "All windows"}} - map place signal`, articlesFor({{time_window:win, subject_l1:area, subject_l2:subj, place_label:p.place_label}}), p.place_count, p.article_count, "mentions", "articles"));
  }});
  drawLegend(places.length ? [["Darker grey = more title-place mentions", "#9a9a9a"], ["Small dots = normalized title place names", "#2f3a46"], ["Use Time window to compare periods", "#64748b"]] : [["No mapped places match the current filters", "#9f3a4a"]]);
}}
function render() {{
  const fixedDiversity = view === "diversity";
  areaEl.disabled = fixedDiversity;
  subjectEl.disabled = fixedDiversity;
  termEl.disabled = !(view === "placeNames" || view === "placeMap");
  timeWindowEl.disabled = !(view === "placeNames" || view === "placeMap" || view === "allSubjects");
  if (fixedDiversity) {{
    areaEl.value = "";
    subjectEl.value = "";
  }}
  refreshSubjectOptions();
  refreshTermOptions();
  if (view === "overview") drawOverview();
  if (view === "subjects") drawSubjects();
  if (view === "allSubjects") drawAllSubjects();
  if (view === "diversity") drawDiversity();
  if (view === "network") drawNetwork();
  if (view === "placeNames") drawPlaceNames();
  if (view === "placeMap") drawPlaceMap();
}}
document.querySelectorAll(".tab").forEach(btn => btn.addEventListener("click", () => {{
  document.querySelectorAll(".tab").forEach(b => b.classList.remove("active"));
  btn.classList.add("active"); view = btn.dataset.view; render();
}}));
areaEl.addEventListener("change", () => {{ subjectEl.value = ""; render(); }});
subjectEl.addEventListener("change", render);
termEl.addEventListener("change", render);
timeWindowEl.addEventListener("change", render);
searchEl.addEventListener("input", () => {{
  panelSub.textContent = "Article list will use the current search filter when you click a chart element.";
}});
window.addEventListener("resize", render);
render();
</script>
</body>
</html>"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_all = load_all_subject_data()
    df = load_data()
    year_min = int(df_all["year_start"].min())
    year_max = int(df_all["year_start"].max())
    year_span = f"{year_min}-{year_max}"
    l1, l2, diversity = build_time_summaries(df)
    full_subject_distribution = build_full_subject_distribution(df_all)
    subject_distribution_by_window = build_subject_distribution_by_window(df_all)
    centrality = build_centrality(df)
    articles = build_articles(df)
    all_articles = build_articles(df_all)
    title_terms = read_optional_csv(TITLE_TERM_TIME_CSV)
    title_bigrams = read_optional_csv(TITLE_BIGRAM_TIME_CSV)
    title_tfidf = read_optional_csv(TITLE_TFIDF_TIME_CSV)
    title_subject_terms = read_optional_csv(TITLE_TERM_SUBJECT_TIME_CSV)
    article_title_terms = read_optional_csv(OUT_DIR / f"article_title_terms_subject_expanded{SUFFIX}.csv")
    title_places = read_optional_csv(TITLE_PLACE_TIME_CSV)
    title_subject_places = read_optional_csv(TITLE_PLACE_SUBJECT_TIME_CSV)
    article_title_places = read_optional_csv(ARTICLE_TITLE_PLACES_CSV)
    title_place_mapping = read_optional_csv(TITLE_PLACE_MAPPING_CSV)

    l1.to_csv(L1_TIME_CSV, index=False, encoding="utf-8")
    l2.to_csv(L2_TIME_CSV, index=False, encoding="utf-8")
    full_subject_distribution.to_csv(FULL_SUBJECT_DIST_CSV, index=False, encoding="utf-8")
    subject_distribution_by_window.to_csv(SUBJECT_DIST_WINDOW_CSV, index=False, encoding="utf-8")
    diversity.to_csv(DIVERSITY_CSV, index=False, encoding="utf-8")
    centrality.to_csv(CENTRALITY_CSV, index=False, encoding="utf-8")

    html = build_html(
        l1,
        l2,
        full_subject_distribution,
        subject_distribution_by_window,
        diversity,
        centrality,
        articles,
        all_articles,
        title_terms,
        title_bigrams,
        title_tfidf,
        title_subject_terms,
        article_title_terms,
        title_places,
        title_subject_places,
        article_title_places,
        title_place_mapping,
        year_span,
    )
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")
    BUNDLE_HTML.write_text(html, encoding="utf-8")

    print(f"rows expanded={len(df)} rows all years={len(df_all)}")
    print(f"l1 rows={len(l1)} l2 rows={len(l2)} all-year subject rows={len(full_subject_distribution)} subject-window rows={len(subject_distribution_by_window)} diversity rows={len(diversity)} centrality rows={len(centrality)}")
    print(f"wrote {OUT_HTML}")
    print(f"wrote {BUNDLE_HTML}")


if __name__ == "__main__":
    main()
