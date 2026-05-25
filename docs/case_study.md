# Case Study: VisitEdinburgh Insight

## Problem

Destination marketing organisations (DMOs) sit on vast quantities of visitor feedback — millions of Airbnb reviews describing what guests loved, struggled with, or ignored entirely. That text is almost never systematically mined. The typical workflow is manual: a marketing analyst reads a sample of reviews, forms impressions, and writes a report. At 473,295 reviews spanning 15 years across Edinburgh's 60+ neighbourhoods, this doesn't scale.

The question this project answers: **can a small, reproducible pipeline make that corpus legible — linking free-text mentions to real places, grouping sentiment into actionable themes, and surfacing the result in a dashboard a non-technical destination manager can actually use?**

---

## Approach: a five-phase pipeline

### Phase 1–2 — Ingest
Raw data comes from two free, no-key-required sources: Inside Airbnb (559,087 reviews + 4,936 listings for Edinburgh, snapshot 21 Sep 2025) and OpenStreetMap via the Overpass API (2,285 local POIs: restaurants, cafes, bars, hotels, museums, attractions). Everything lands in a single DuckDB file — chosen for its SQL semantics, single-file portability, and ability to read CSV/Parquet natively with no server overhead.

### Phase 3 — Clean & deduplicate
POI names are lowercased, punctuation stripped, and abbreviations expanded (`St` → `Street`). POIs within 50m with the same normalised name collapse to a single canonical row, reducing the set from 2,285 to 2,270. Reviews are filtered for English (via `langdetect`) and minimum length (10+ characters), trimming 15.35% of raw rows — mostly one-line non-English entries — down to 473,295 clean reviews. Each review inherits a neighbourhood label via spatial join from its listing's lat/lon.

### Phase 4 — Geocode: linking reviews to places
This is the core accuracy challenge. A naive substring search (`str.contains`) matched 541,513 review-POI pairs — but inspection showed catastrophic false positives: a cafe named "Eve" matched "eventually", "home" matched "home away from home" without any geographic intent, and "the place" and "the street" each matched tens of thousands of generic sentences.

The fix was three-layered:
1. **Minimum name length** of 5 characters drops names too short to be distinctive.
2. **Word-boundary regex** (`\b…\b`) prevents suffix and prefix matches.
3. **Blocklist** of 36 common English words and phrases (including multi-word OSM venue names like "the place" and "the street") that still produce false positives after boundary matching.

After the fix, exact matching produces 29,053 high-precision links. A second NER+fuzzy pass — running spaCy's `en_core_web_sm` NER over every review, then fuzzy-matching each GPE/FAC/ORG/LOC entity against POIs within 1km of the listing — adds a further 152,671 links, for **181,724 total (5.3× uplift)**. The NER pass catches cases like "Grass Market" (two words) matching "grassmarket" (one word) with a rapidfuzz score of ~96.

The bug was caught via the dashboard itself: the Top-10 POIs table showed "the place" (a cafe) with 29,168 mentions — implausibly more than Edinburgh's most-visited attractions. No automated test would have caught this without the visual sanity check.

### Phase 5 — Theme analysis
473,295 reviews are vectorised with `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional embeddings, cached to disk). KMeans clustering at k=8 (chosen by informal silhouette inspection) produces 8 clusters of ~59K reviews each. Clusters are labelled by comparative TF-IDF: for each cluster, terms are scored by `cluster_prevalence × (cluster_frequency / global_frequency)`, with a 3% minimum cluster-prevalence gate to suppress rare noise words. This yields labels like:

- `town · centre · mile · royal · restaurants` — Old Town sightseeing cluster
- `room · kitchen · quiet · bathroom · need` — self-catering amenities cluster
- `friendly · helpful · responsive · amazing · room` — host quality cluster
- `quiet · centre · helpful · definitely · located` — location satisfaction cluster

### Phase 6 — Dashboard
A three-tab Streamlit app backed entirely by DuckDB reads (no recomputation on load, `@st.cache_data` throughout):

- **Overview** — KPI strip (473K reviews, 2,270 POIs, Aug 2010 – Sep 2025, 2,600/month), PyDeck scatter map with log-radius scaling and Okabe-Ito colorblind palette, Top-10 POI table.
- **Themes** — 8-band stacked area chart showing theme share over time (COVID-19 dip visible as a gap in 2020–21), top-theme-per-neighbourhood horizontal bar, 3-review sample selector.
- **Recommendations** — 6 rule-based insight cards with evidence numbers and neighbourhood pins (e.g. rising-theme detection, self-catering hotspot, sightseeing flow signal).

---

## Results

| Metric | Value |
|---|---|
| Reviews processed | 473,295 |
| POI records | 2,270 |
| Review-to-POI links (exact only) | 29,053 |
| Review-to-POI links (with NER+fuzzy) | **181,724** |
| Location-match uplift | **5.3×** |
| Visitor themes | **8** |

**Qualitative finding:** The self-catering amenities cluster (`room · kitchen · quiet · bathroom`) over-indexes significantly in several Edinburgh neighbourhoods — a concrete signal for DMOs to prioritise kitchen-equipped property marketing in those areas. The Old Town sightseeing cluster dominates Holyrood and Old Town neighbourhoods at well above city-average share, suggesting visitor-flow management conversations with attractions operators.

---

## Reflection

The most instructive moment in the build was the Phase 4 false-positive bug. The initial implementation used pandas `str.contains(regex=False)` — fast, correct for substring search, but with no word boundaries. It produced 541K links and looked plausible until the dashboard's Top-10 table surfaced "the place" and "the street" as Edinburgh's most-mentioned venues. That's a signal no unit test caught, because the tests were written against the same flawed assumptions.

The fix — word boundaries, minimum length, blocklist — required three iterations because each pass exposed a new category of false positive. The final exact count (29,053) is much lower than the naive figure, but the high NER+fuzzy uplift (5.3×) means total coverage is actually higher than the inflated original, and every link in the table now has a traceable, defensible reason for being there.

The lesson: for any NLP pipeline that writes to a user-facing table, the most valuable test is "does the top-10 list make sense to a human who knows the domain?" A recruiter or destination manager seeing "the place" at #1 would correctly conclude the system is broken — and they'd be right.
