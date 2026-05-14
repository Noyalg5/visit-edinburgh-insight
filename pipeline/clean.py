"""Phase 3 — normalise POIs, dedupe, filter reviews, assign neighbourhoods."""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path

import duckdb
import pandas as pd
from langdetect import DetectorFactory, LangDetectException, detect
from shapely.geometry import Point, shape

from pipeline import metrics as metrics_module
from pipeline.config import DATA_RAW_DIR, DB_PATH

DetectorFactory.seed = 0  # reproducible language detection

log = logging.getLogger(__name__)

_GEO_PATH = DATA_RAW_DIR / "insideairbnb" / "neighbourhoods.geojson"

# Applied before lowercasing so case-sensitive boundary matches work
_ABBREVS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bSt\b", re.IGNORECASE), "Street"),
    (re.compile(r"\bRd\b", re.IGNORECASE), "Road"),
    (re.compile(r"\bAve\b", re.IGNORECASE), "Avenue"),
]
_PUNCT = re.compile(r"[^\w\s]")


# ---------------------------------------------------------------------------
# POI helpers
# ---------------------------------------------------------------------------

def _normalise_name(name: str | None) -> str:
    """Expand abbreviations, lowercase, strip punctuation."""
    if not name:
        return ""
    s = str(name)
    for pat, repl in _ABBREVS:
        s = pat.sub(repl, s)
    s = s.lower()
    s = _PUNCT.sub("", s)
    return " ".join(s.split())


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _dedupe_pois(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add canonical_name, then collapse rows sharing the same canonical_name
    AND within 50 m into one clean_pois row; preserve source_osm_ids list.
    """
    df = df.copy()
    df["canonical_name"] = df["name"].apply(_normalise_name)
    result_rows: list[dict] = []

    for canonical_name, group in df.groupby("canonical_name", sort=False):
        records = group.reset_index(drop=True)

        if len(records) == 1:
            r = records.iloc[0]
            result_rows.append({
                "poi_id":         int(r["osm_id"]),
                "canonical_name": canonical_name,
                "category":       r["category"],
                "latitude":       float(r["latitude"]),
                "longitude":      float(r["longitude"]),
                "source_osm_ids": [int(r["osm_id"])],
                "neighbourhood":  None,
            })
            continue

        used: set[int] = set()
        for i in range(len(records)):
            if i in used:
                continue
            ri = records.iloc[i]
            cluster = [i]
            for j in range(i + 1, len(records)):
                if j in used:
                    continue
                rj = records.iloc[j]
                if _haversine_m(
                    float(ri["latitude"]), float(ri["longitude"]),
                    float(rj["latitude"]), float(rj["longitude"]),
                ) <= 50.0:
                    cluster.append(j)
            for idx in cluster:
                used.add(idx)
            rep = records.iloc[cluster[0]]
            result_rows.append({
                "poi_id":         int(rep["osm_id"]),
                "canonical_name": canonical_name,
                "category":       rep["category"],
                "latitude":       float(rep["latitude"]),
                "longitude":      float(rep["longitude"]),
                "source_osm_ids": [int(records.iloc[idx]["osm_id"]) for idx in cluster],
                "neighbourhood":  None,
            })

    return pd.DataFrame(result_rows)


# ---------------------------------------------------------------------------
# Geography helpers
# ---------------------------------------------------------------------------

def _load_polygons(geo_path: Path) -> list[tuple[str, object]]:
    """Load neighbourhoods.geojson → [(name, Shapely polygon), …]."""
    features = json.loads(geo_path.read_text(encoding="utf-8"))["features"]
    return [
        (f["properties"]["neighbourhood"], shape(f["geometry"]))
        for f in features
        if f.get("geometry") is not None
    ]


def _find_neighbourhood(lat: float, lon: float, polygons: list[tuple[str, object]]) -> str | None:
    """Point-in-polygon lookup; returns first matching neighbourhood name."""
    pt = Point(lon, lat)  # Shapely convention: (x=lon, y=lat)
    for name, poly in polygons:
        if poly.contains(pt):
            return name
    return None


# ---------------------------------------------------------------------------
# Review helpers
# ---------------------------------------------------------------------------

def _detect_lang(text: str) -> str:
    try:
        return detect(str(text))
    except LangDetectException:
        return "unknown"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def _create_clean_tables(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("DROP TABLE IF EXISTS clean_pois")
    conn.execute("""
        CREATE TABLE clean_pois (
            poi_id         BIGINT PRIMARY KEY,
            canonical_name TEXT,
            category       TEXT,
            latitude       DOUBLE,
            longitude      DOUBLE,
            source_osm_ids BIGINT[],
            neighbourhood  TEXT
        )
    """)
    conn.execute("DROP TABLE IF EXISTS clean_reviews")
    conn.execute("""
        CREATE TABLE clean_reviews (
            review_id     BIGINT PRIMARY KEY,
            listing_id    BIGINT,
            review_date   DATE,
            comment_clean TEXT,
            lang          TEXT,
            char_count    INT,
            neighbourhood TEXT
        )
    """)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run() -> dict:
    conn = duckdb.connect(str(DB_PATH))
    try:
        raw_pois     = conn.execute("SELECT * FROM raw_pois").df()
        raw_reviews  = conn.execute("SELECT * FROM raw_reviews").df()
        raw_listings = conn.execute(
            "SELECT listing_id, latitude, longitude FROM raw_listings"
        ).df()

        # ── POIs ──────────────────────────────────────────────────────────
        log.info("Normalising and deduplicating %d raw POIs …", len(raw_pois))
        clean_pois_df = _dedupe_pois(raw_pois)

        polygons = _load_polygons(_GEO_PATH)
        log.info("Loaded %d neighbourhood polygons", len(polygons))

        clean_pois_df["neighbourhood"] = [
            _find_neighbourhood(row["latitude"], row["longitude"], polygons)
            for _, row in clean_pois_df.iterrows()
        ]
        log.info("clean_pois: %d rows (from %d raw)", len(clean_pois_df), len(raw_pois))

        # ── Reviews ───────────────────────────────────────────────────────
        log.info("Cleaning %d raw reviews …", len(raw_reviews))
        df = raw_reviews.copy()
        df["review_date"]   = pd.to_datetime(df["review_date"], errors="coerce")
        df["comment_clean"] = df["comment"].fillna("").str.strip()
        df["char_count"]    = df["comment_clean"].str.len()

        n_raw = len(df)
        df = df[df["char_count"] >= 10].copy()
        log.info("Dropped %d reviews with char_count < 10 (%d remain)", n_raw - len(df), len(df))

        log.info("Detecting language for %d reviews — this takes several minutes …", len(df))
        langs: list[str] = []
        comments = df["comment_clean"].tolist()
        for i, comment in enumerate(comments):
            if i % 10_000 == 0 and i > 0:
                log.info("  … language detection %d / %d", i, len(comments))
            langs.append(_detect_lang(comment))
        df["lang"] = langs

        n_after_char = len(df)
        df = df[df["lang"] == "en"].copy()
        log.info(
            "Dropped %d non-English reviews (%d remain)",
            n_after_char - len(df), len(df),
        )

        # neighbourhood: look up each listing's lat/lon once, then map to reviews
        log.info("Assigning neighbourhoods to %d listings …", len(raw_listings))
        listing_nbhd: dict[int, str | None] = {
            int(row["listing_id"]): _find_neighbourhood(
                float(row["latitude"]), float(row["longitude"]), polygons
            )
            for _, row in raw_listings.iterrows()
        }
        df["neighbourhood"] = df["listing_id"].map(listing_nbhd)

        clean_reviews_df = df[[
            "review_id", "listing_id", "review_date",
            "comment_clean", "lang", "char_count", "neighbourhood",
        ]].copy()

        # ── Write ─────────────────────────────────────────────────────────
        _create_clean_tables(conn)
        conn.execute("INSERT INTO clean_pois    SELECT * FROM clean_pois_df")
        conn.execute("INSERT INTO clean_reviews SELECT * FROM clean_reviews_df")
        log.info(
            "Wrote %d clean_pois and %d clean_reviews to %s",
            len(clean_pois_df), len(clean_reviews_df), DB_PATH,
        )

        # ── Metrics ───────────────────────────────────────────────────────
        m: dict = {
            "raw_pois_count":             int(len(raw_pois)),
            "clean_pois_count":           int(len(clean_pois_df)),
            "poi_duplicate_reduction_pct": round(
                (len(raw_pois) - len(clean_pois_df)) / len(raw_pois) * 100, 2
            ),
            "raw_reviews_count":          int(len(raw_reviews)),
            "clean_reviews_count":        int(len(clean_reviews_df)),
            "review_filter_drop_pct":     round(
                (len(raw_reviews) - len(clean_reviews_df)) / len(raw_reviews) * 100, 2
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
    log.info("Clean complete. Metrics:\n%s", json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
