# PROJECT_SPEC.md — VisitEdinburgh Insight

> Single source of truth for the project. When prompting Claude Code, refer to this file ("per PROJECT_SPEC.md…"). Update it as decisions evolve; do not let scope drift live only in chat.

## 1. Elevator pitch

A small-scale insight platform that turns visitor reviews and local business data for **Edinburgh, Scotland** into actionable intelligence for destination managers and city marketers.

The deliverable is:

1. A reproducible Python pipeline (ingest → clean → geocode → analyse) backed by DuckDB.
2. A Streamlit dashboard that a non-technical destination manager could navigate.
3. A short case-study README that quotes concrete before/after metrics.

This is a small-scale reproduction of a final-year university project. Scope is intentionally constrained to a single city and a single review source to keep build time to ~1–2 days while still demonstrating end-to-end data engineering, analysis, and product thinking.

## 2. Success metrics (compute these from real run logs)

All three are headline claims in the README and CV bullet. They must be derived programmatically and logged to `pipeline/metrics.json` on each run, not hand-edited.

- **Duplicate reduction (%)** — `(raw_rows - deduped_rows) / raw_rows * 100`, reported separately for POIs and reviews. Target: ≥ 25% on POIs.
- **Location-match accuracy uplift (%)** — `(fuzzy_match_count - exact_match_count) / exact_match_count * 100`, where a "match" means a review's free-text place mention was successfully linked to a row in the POI table. Target: ≥ 30% uplift.
- **Manual review time reduction** — proxy metric: count of records flagged for human attention before vs. after applying the cleaning rules. Express as "X% fewer rows require manual review."

## 3. Data sources (free, real, no API keys required)

### 3.1 Inside Airbnb — visitor feedback
- Site: http://insideairbnb.com/get-the-data
- Files needed for Edinburgh (latest snapshot available):
  - `listings.csv.gz` — listing-level data with lat/lon, neighbourhood.
  - `reviews.csv.gz` — every public review with date, listing_id, reviewer_id, comment text.
- Robots/terms: the site explicitly offers these files for non-commercial research. Keep a copy in `data/raw/insideairbnb/` and cite in the README.

### 3.2 OpenStreetMap via Overpass API — local business / POI reference
- Endpoint: `https://overpass-api.de/api/interpreter`
- Query (bounding box covers central Edinburgh):

```overpassql
[out:json][timeout:60];
(
  node["amenity"~"restaurant|cafe|bar|pub|hotel|fast_food"](55.92,-3.25,55.98,-3.13);
  node["tourism"~"attraction|museum|gallery|viewpoint|hotel"](55.92,-3.25,55.98,-3.13);
);
out body;
```

Cache the JSON response in `data/raw/osm/edinburgh_pois.json`. Refresh on `--refresh` flag only.

### 3.3 Edinburgh neighbourhood boundaries (optional but recommended)
- Source: https://data.spatialhub.scot/ (search "data zones" for Edinburgh) OR fall back to Inside Airbnb's `neighbourhoods.geojson` which is included in the listings download.
- Used purely for the dashboard map overlay and "filter by neighbourhood" UX.

## 4. Storage & schema — DuckDB

DuckDB chosen because: SQL semantics, single-file storage, reads Parquet/CSV natively, no server. Database file: `db/visit.duckdb`.

### 4.1 Tables

```sql
-- raw_reviews: 1 row per review from Inside Airbnb
CREATE TABLE raw_reviews (
  review_id BIGINT PRIMARY KEY,
  listing_id BIGINT,
  review_date DATE,
  reviewer_id BIGINT,
  reviewer_name TEXT,
  comment TEXT
);

-- raw_listings: 1 row per Airbnb listing
CREATE TABLE raw_listings (
  listing_id BIGINT PRIMARY KEY,
  name TEXT,
  neighbourhood TEXT,
  latitude DOUBLE,
  longitude DOUBLE,
  room_type TEXT,
  price DOUBLE
);

-- raw_pois: 1 row per OSM amenity/tourism node
CREATE TABLE raw_pois (
  osm_id BIGINT PRIMARY KEY,
  name TEXT,
  category TEXT,          -- restaurant, cafe, museum, etc.
  latitude DOUBLE,
  longitude DOUBLE,
  tags JSON
);

-- clean_pois: deduped + normalised POIs
CREATE TABLE clean_pois (
  poi_id BIGINT PRIMARY KEY,    -- generated
  canonical_name TEXT,
  category TEXT,
  latitude DOUBLE,
  longitude DOUBLE,
  source_osm_ids BIGINT[],      -- which raw rows collapsed into this one
  neighbourhood TEXT
);

-- clean_reviews: deduped + cleaned reviews with derived fields
CREATE TABLE clean_reviews (
  review_id BIGINT PRIMARY KEY,
  listing_id BIGINT,
  review_date DATE,
  comment_clean TEXT,
  lang TEXT,
  char_count INT,
  neighbourhood TEXT
);

-- review_poi_links: which POIs a review mentions
CREATE TABLE review_poi_links (
  review_id BIGINT,
  poi_id BIGINT,
  match_method TEXT,            -- 'exact' | 'fuzzy' | 'ner'
  confidence DOUBLE,
  PRIMARY KEY (review_id, poi_id)
);

-- review_themes: topic cluster per review
CREATE TABLE review_themes (
  review_id BIGINT,
  theme_id INT,
  theme_label TEXT,
  weight DOUBLE
);
```

## 5. Architecture / repo layout

```
visit-edinburgh-insight/
├── PROJECT_SPEC.md           # this file
├── README.md                 # written last, includes screenshots + metrics
├── requirements.txt
├── .gitignore                # ignore data/, db/, .venv/
├── data/
│   ├── raw/                  # never committed
│   └── interim/
├── db/
│   └── visit.duckdb          # never committed
├── pipeline/
│   ├── __init__.py
│   ├── ingest.py             # Phase 2
│   ├── clean.py              # Phase 3
│   ├── geocode.py            # Phase 4
│   ├── analyse.py            # Phase 5
│   ├── metrics.py            # writes metrics.json
│   └── config.py             # paths, bounding boxes, constants
├── app/
│   └── dashboard.py          # Phase 6 — Streamlit
├── notebooks/
│   └── 01_exploration.ipynb  # scratchpad, kept light
├── tests/
│   ├── test_clean.py
│   ├── test_geocode.py
│   └── fixtures/
├── docs/
│   ├── screenshots/
│   └── case_study.md         # one-pager for portfolio
└── pipeline/metrics.json     # auto-generated, COMMITTED
```

## 6. Phase-by-phase deliverables

Each phase ends with a git commit and updated `metrics.json` where relevant. Phases are run in order; do not parallelise.

### Phase 1 — Scaffold (≈ 30 min)
- Repo structure above.
- `requirements.txt` pinned to: `duckdb`, `pandas`, `pyarrow`, `requests`, `rapidfuzz`, `spacy` (with `en_core_web_sm`), `scikit-learn`, `sentence-transformers`, `streamlit`, `pydeck`, `plotly`, `pytest`, `python-dotenv`.
- `.gitignore` covering `data/`, `db/`, `.venv/`, `.streamlit/`, `__pycache__/`.
- Stub README with project title and "build instructions coming."

### Phase 2 — Ingest (≈ 1 h)
- `pipeline/ingest.py` with CLI `python -m pipeline.ingest [--refresh]`.
- Downloads Inside Airbnb files and Overpass POI dump. Caches raw files; skips download if cache exists unless `--refresh`.
- Loads into `raw_reviews`, `raw_listings`, `raw_pois`.
- Logs row counts.

### Phase 3 — Clean & dedupe (≈ 2 h) — produces metric #1
- `pipeline/clean.py`:
  - Normalise POI names: lowercase, strip punctuation, expand abbreviations (`St` → `Street`, `Rd` → `Road`).
  - Collapse POIs with same normalised name + within 50m → single `clean_pois` row.
  - Drop reviews with `char_count < 10` or `lang != 'en'` (use a fast detector like `langdetect`).
  - Parse `review_date`, derive `neighbourhood` for each review by spatial join to listing → neighbourhood boundary.
- Writes before/after counts to `metrics.json`.

### Phase 4 — Geocode / location match (≈ 2 h) — produces metric #2
- `pipeline/geocode.py`:
  - **Baseline:** exact substring match of POI canonical_name in each `comment_clean`. Record matches with `match_method='exact'`.
  - **Improved:** spaCy NER (`GPE`, `FAC`, `ORG`, `LOC` entities) → for each entity, fuzzy match (`rapidfuzz.process.extractOne`, score_cutoff=85) against POI names within 1km of the listing's lat/lon. Record with `match_method='fuzzy'` or `'ner'`.
  - Compare match counts; write uplift % to `metrics.json`.

### Phase 5 — Analyse themes (≈ 2 h)
- `pipeline/analyse.py`:
  - Vectorise `comment_clean` with `sentence-transformers/all-MiniLM-L6-v2`.
  - KMeans (`k=8`) → assign cluster to each review.
  - Label clusters by top TF-IDF terms in each cluster (e.g. "queues & wait times", "value for money", "scenic views").
  - Aggregate: top theme by neighbourhood, theme trend over time, mention volume per POI.

### Phase 6 — Dashboard (≈ 3 h)
- `app/dashboard.py` — Streamlit, three tabs:

  **Tab 1: Overview**
  - KPI strip: total reviews, total POIs, date range, avg reviews/month.
  - PyDeck map of Edinburgh with POIs coloured by category and sized by mention count.
  - Sidebar filters: date range, neighbourhood (multiselect), POI category.

  **Tab 2: Themes**
  - Stacked area chart: theme volume over time.
  - Bar chart: top themes per selected neighbourhood.
  - Sample review text for the selected theme (3 examples, anonymised — show listing_id only).

  **Tab 3: Recommendations**
  - Rule-based cards. Each card = one suggestion + the evidence behind it. Example rule:
    - IF theme = "queues" AND share of mentions in last 90 days > 20% → "Consider off-peak campaign for [neighbourhood]; queue mentions up vs. baseline."
  - 5–8 such rules; keep them simple and traceable.

- Reads everything from `db/visit.duckdb` — no recomputation on page load. Cache with `@st.cache_data`.

### Phase 7 — Polish (≈ 2 h)
- Take 3 screenshots into `docs/screenshots/`.
- Write README: pitch, architecture diagram (mermaid), how to run, metrics table, screenshots, "what I'd do next."
- Write `docs/case_study.md`: one page, problem → approach → results → reflection.
- Deploy to **Streamlit Community Cloud** (free): connect GitHub repo, point at `app/dashboard.py`. Add the live URL to README.
- Final CV bullet draft with real numbers.

## 7. Dashboard wireframe (ASCII)

```
┌─ VisitEdinburgh Insight ─────────────────────────────────────┐
│  [Overview] [Themes] [Recommendations]                       │
├──────────────┬───────────────────────────────────────────────┤
│ FILTERS      │  Reviews: 48,213    POIs: 1,847               │
│              │  Date: 2018–2024    Avg/mo: 612               │
│ Date range   │                                               │
│ [────●──●──] │  ┌─────────────────────────────────────────┐  │
│              │  │                                         │  │
│ Neighbourhood│  │         Edinburgh map (PyDeck)          │  │
│ ☑ Old Town   │  │      POIs sized by mention count        │  │
│ ☑ New Town   │  │                                         │  │
│ ☐ Leith      │  └─────────────────────────────────────────┘  │
│ ☐ Stockbridge│                                               │
│              │  Top mentioned POIs (selected area):          │
│ Category     │  1. Edinburgh Castle     1,204 mentions       │
│ ☑ Attraction │  2. Royal Mile             892 mentions       │
│ ☑ Restaurant │  3. Arthur's Seat          611 mentions       │
└──────────────┴───────────────────────────────────────────────┘
```

## 8. Coding standards

- Python 3.11+.
- Type hints on all public functions.
- No notebooks in the pipeline path — notebooks are exploration only.
- One pytest happy-path test per pipeline module, using small fixture CSVs in `tests/fixtures/`.
- Functions over classes unless state is genuinely needed.
- Log with the `logging` module, not `print`.
- Constants (bounding boxes, file URLs, thresholds) live in `pipeline/config.py`, not scattered.

## 9. Out of scope (do not let Claude Code add these)

- Authentication / user accounts.
- Multiple cities.
- LLM-based summarisation (keep theme labelling reproducible).
- Real-time data refresh / scheduling.
- Docker / Kubernetes / Terraform.
- A custom React frontend.

If Claude Code suggests any of the above, push back and point at this section.

## 10. Portfolio polish checklist

- [ ] Repo is public on GitHub with descriptive name.
- [ ] README opens with a 3-bullet summary and a hero screenshot.
- [ ] Mermaid architecture diagram in README.
- [ ] `metrics.json` values quoted in README (sourced from real run, not invented).
- [ ] Live Streamlit Cloud URL in README.
- [ ] LinkedIn post draft (optional) referencing the live demo.
- [ ] CV bullet rewritten using real numbers from `metrics.json`.

## 11. CV bullet template (fill from real run)

> Built a Python + DuckDB pipeline and Streamlit dashboard that ingested ~{N} Edinburgh visitor reviews and ~{M} OSM POIs, reduced duplicate POI records by **{X}%**, improved free-text location-match accuracy by **{Y}%** via spaCy NER + fuzzy matching, and surfaced neighbourhood-level theme trends for destination-marketing recommendations. Deployed on Streamlit Cloud — [live link].

---

*Use this spec verbatim with Claude Code. Begin by saying: "Read PROJECT_SPEC.md. Execute Phase 1 only."*
