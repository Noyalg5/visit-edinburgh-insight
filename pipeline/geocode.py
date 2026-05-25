"""Phase 4 — link reviews to POIs via exact substring and NER+fuzzy matching."""

from __future__ import annotations

import json
import logging
import re

import duckdb
import numpy as np
import pandas as pd
import spacy
from rapidfuzz.process import extractOne

from pipeline import metrics as metrics_module
from pipeline.config import DB_PATH

log = logging.getLogger(__name__)

# spaCy entity labels considered as place/venue mentions
_NER_LABELS = {"GPE", "FAC", "ORG", "LOC"}

# Components to skip — keep only tok2vec + ner for speed
_DISABLE = ["tagger", "parser", "senter", "attribute_ruler", "lemmatizer"]


# ---------------------------------------------------------------------------
# Exact match
# ---------------------------------------------------------------------------

# Short or generic English words that produce massive false-positive rates
# even with word-boundary matching (e.g. "close" in "it was close to the
# station", "park" in "we parked nearby").  Names in this set are skipped
# for exact matching but remain eligible for NER+fuzzy if spaCy tags them
# as a proper-noun entity in context.
_EXACT_BLOCKLIST: frozenset[str] = frozenset({
    "eve", "che", "close", "home", "house", "place", "world", "store",
    "shop", "front", "south", "north", "east", "west", "garden", "park",
    "gate", "lane", "road", "street", "bridge", "court", "hall", "hill",
    "view", "rise", "walk", "mill", "bank", "green", "grove", "vale",
    "lodge", "manor", "mews", "point", "ridge", "mount", "field",
})

_MIN_EXACT_LEN = 5  # skip POI names shorter than this


def _exact_match(
    reviews_df: pd.DataFrame,
    name_to_id: dict[str, int],
) -> list[dict]:
    """
    Word-boundary exact match: for each POI canonical name that passes the
    length and blocklist filters, use \\b…\\b regex so "close" cannot match
    "closely" and "home" cannot match "homemade".
    Longer names first; a seen-set deduplicates (review_id, poi_id) pairs.
    """
    comments   = reviews_df["comment_clean"].str.lower().fillna("")
    review_ids = reviews_df["review_id"].to_numpy()

    seen:  set[tuple[int, int]] = set()
    links: list[dict]           = []

    for poi_name, poi_id in sorted(name_to_id.items(), key=lambda x: len(x[0]), reverse=True):
        if not poi_name:
            continue
        if len(poi_name) < _MIN_EXACT_LEN:
            continue
        if poi_name in _EXACT_BLOCKLIST:
            continue

        pattern = rf"\b{re.escape(poi_name)}\b"
        mask = comments.str.contains(pattern, regex=True, case=False, na=False)
        for rev_id in review_ids[mask.to_numpy()]:
            key = (int(rev_id), int(poi_id))
            if key in seen:
                continue
            seen.add(key)
            links.append({
                "review_id":    int(rev_id),
                "poi_id":       int(poi_id),
                "match_method": "exact",
                "confidence":   1.0,
            })
    return links


# ---------------------------------------------------------------------------
# NER + fuzzy match
# ---------------------------------------------------------------------------

def _pois_near(
    lat: float,
    lon: float,
    poi_lats: np.ndarray,
    poi_lons: np.ndarray,
    poi_ids: np.ndarray,
    poi_names: np.ndarray,
    radius_m: float = 1000.0,
) -> list[tuple[int, str]]:
    """Vectorised haversine — returns [(poi_id, canonical_name)] within radius_m."""
    phi1  = np.radians(lat)
    dphi  = np.radians(poi_lats - lat)
    dlam  = np.radians(poi_lons - lon)
    a     = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(np.radians(poi_lats)) * np.sin(dlam / 2) ** 2
    dist  = 2 * 6_371_000 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    mask  = dist <= radius_m
    return list(zip(poi_ids[mask].tolist(), poi_names[mask].tolist()))


def _ner_fuzzy_match(
    reviews_df: pd.DataFrame,
    clean_pois_df: pd.DataFrame,
    listing_loc: dict[int, tuple[float, float]],
    nlp,
    existing_pairs: set[tuple[int, int]],
    batch_size: int = 256,
) -> list[dict]:
    """
    Run spaCy NER on every review; for each GPE/FAC/ORG/LOC entity,
    rapidfuzz-match against POIs within 1 km of the listing (score_cutoff=85).
    Returns only links NOT already in existing_pairs.
    """
    poi_lats  = clean_pois_df["latitude"].to_numpy()
    poi_lons  = clean_pois_df["longitude"].to_numpy()
    poi_ids   = clean_pois_df["poi_id"].to_numpy()
    poi_names = clean_pois_df["canonical_name"].to_numpy()

    # Pre-compute per-listing nearby POI list (avoids repeat vectorised calls)
    listing_nearby: dict[int, list[tuple[int, str]]] = {
        lid: _pois_near(lat, lon, poi_lats, poi_lons, poi_ids, poi_names)
        for lid, (lat, lon) in listing_loc.items()
    }

    records = reviews_df[["review_id", "listing_id", "comment_clean"]].to_dict("records")
    n = len(records)
    links: list[dict] = []
    seen = set(existing_pairs)

    for batch_start in range(0, n, batch_size):
        if batch_start > 0 and batch_start % 50_000 == 0:
            log.info("  … NER+fuzzy %d / %d reviews", batch_start, n)

        batch   = records[batch_start: batch_start + batch_size]
        texts   = [str(r["comment_clean"]) for r in batch]

        for doc, rec in zip(nlp.pipe(texts, disable=_DISABLE, batch_size=batch_size), batch):
            review_id  = int(rec["review_id"])
            nearby     = listing_nearby.get(int(rec["listing_id"]), [])
            if not nearby:
                continue

            nearby_names = [name for _, name in nearby]
            name_to_pid  = {name: pid for pid, name in nearby}

            for ent in doc.ents:
                if ent.label_ not in _NER_LABELS:
                    continue

                result = extractOne(ent.text.lower(), nearby_names, score_cutoff=85)
                if result is None:
                    continue

                matched_name, score, _ = result
                poi_id = name_to_pid[matched_name]
                key    = (review_id, poi_id)
                if key in seen:
                    continue
                seen.add(key)
                links.append({
                    "review_id":    review_id,
                    "poi_id":       poi_id,
                    "match_method": "ner",
                    "confidence":   round(score / 100.0, 4),
                })

    return links


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def _create_links_table(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("DROP TABLE IF EXISTS review_poi_links")
    conn.execute("""
        CREATE TABLE review_poi_links (
            review_id    BIGINT,
            poi_id       BIGINT,
            match_method TEXT,
            confidence   DOUBLE,
            PRIMARY KEY (review_id, poi_id)
        )
    """)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run() -> dict:
    conn = duckdb.connect(str(DB_PATH))
    try:
        clean_reviews_df = conn.execute(
            "SELECT review_id, listing_id, comment_clean FROM clean_reviews"
        ).df()
        clean_pois_df = conn.execute(
            "SELECT poi_id, canonical_name, latitude, longitude FROM clean_pois"
        ).df()
        listings_df = conn.execute(
            "SELECT listing_id, latitude, longitude FROM raw_listings"
        ).df()

        name_to_id: dict[str, int] = dict(
            zip(clean_pois_df["canonical_name"], clean_pois_df["poi_id"].astype(int))
        )
        listing_loc: dict[int, tuple[float, float]] = {
            int(r.listing_id): (float(r.latitude), float(r.longitude))
            for r in listings_df.itertuples(index=False)
        }

        # ── Exact pass ────────────────────────────────────────────────────
        log.info("Exact-match pass over %d reviews …", len(clean_reviews_df))
        exact_links = _exact_match(clean_reviews_df, name_to_id)
        log.info("Exact pass: %d review-POI links", len(exact_links))

        # ── NER+fuzzy pass ────────────────────────────────────────────────
        log.info("Loading spaCy en_core_web_sm …")
        nlp = spacy.load("en_core_web_sm")

        existing_pairs = {(d["review_id"], d["poi_id"]) for d in exact_links}
        log.info(
            "NER+fuzzy pass over %d reviews (pre-computed %d listing radii) …",
            len(clean_reviews_df), len(listing_loc),
        )
        ner_links = _ner_fuzzy_match(
            clean_reviews_df, clean_pois_df, listing_loc, nlp, existing_pairs
        )
        log.info("NER pass: %d additional review-POI links", len(ner_links))

        # ── Write ─────────────────────────────────────────────────────────
        all_links  = exact_links + ner_links
        links_df   = pd.DataFrame(all_links) if all_links else pd.DataFrame(
            columns=["review_id", "poi_id", "match_method", "confidence"]
        )
        _create_links_table(conn)
        if len(links_df):
            conn.execute("INSERT INTO review_poi_links SELECT * FROM links_df")
        log.info("review_poi_links: %d total rows written", len(links_df))

        # ── Metrics ───────────────────────────────────────────────────────
        exact_count = len(exact_links)
        fuzzy_count = exact_count + len(ner_links)
        m: dict = {
            "exact_match_count":        exact_count,
            "fuzzy_match_count":        fuzzy_count,
            "location_match_uplift_pct": round(
                (fuzzy_count - exact_count) / max(exact_count, 1) * 100, 2
            ),
        }
        metrics_module.write(m)
        return m

    finally:
        conn.close()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    m = run()
    log.info("Geocode complete. Metrics:\n%s", json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
