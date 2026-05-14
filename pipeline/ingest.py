"""Phase 2 — download raw data and load into DuckDB."""

import argparse
import json
import logging
from pathlib import Path

import duckdb
import pandas as pd
import requests

from pipeline.config import (
    DATA_RAW_DIR,
    DB_PATH,
    EDINBURGH_BBOX,
    INSIDE_AIRBNB_LISTINGS_URL,
    INSIDE_AIRBNB_NEIGHBOURHOODS_URL,
    INSIDE_AIRBNB_REVIEWS_URL,
    OVERPASS_URL,
)

log = logging.getLogger(__name__)

_AIRBNB_DIR = DATA_RAW_DIR / "insideairbnb"
_OSM_DIR = DATA_RAW_DIR / "osm"

_s, _w, _n, _e = EDINBURGH_BBOX
_OVERPASS_QUERY = f"""\
[out:json][timeout:60];
(
  node["amenity"~"restaurant|cafe|bar|pub|hotel|fast_food"]({_s},{_w},{_n},{_e});
  node["tourism"~"attraction|museum|gallery|viewpoint|hotel"]({_s},{_w},{_n},{_e});
);
out body;
"""

# Columns we want from the Inside Airbnb listings file (it has ~70 columns total)
_LISTING_COLS = {
    "id", "name", "neighbourhood_cleansed", "neighbourhood",
    "latitude", "longitude", "room_type", "price",
}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download(url: str, dest: Path, refresh: bool) -> None:
    """Download url to dest, skipping if dest exists and refresh is False."""
    if dest.exists() and not refresh:
        log.info("Cache hit — skipping download: %s", dest.name)
        return
    log.info("Downloading %s", url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    log.info("Saved %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)


def _fetch_overpass(dest: Path, refresh: bool) -> None:
    """POST Overpass query and cache the JSON response."""
    if dest.exists() and not refresh:
        log.info("Cache hit — skipping Overpass query: %s", dest.name)
        return
    log.info("Querying Overpass API …")
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.post(
        OVERPASS_URL,
        data={"data": _OVERPASS_QUERY},
        headers={"Accept": "*/*"},
        timeout=90,
    )
    resp.raise_for_status()
    dest.write_text(resp.text, encoding="utf-8")
    n = len(resp.json().get("elements", []))
    log.info("Overpass returned %d elements → %s", n, dest.name)


# ---------------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------------

def _create_tables(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_listings (
            listing_id    BIGINT PRIMARY KEY,
            name          TEXT,
            neighbourhood TEXT,
            latitude      DOUBLE,
            longitude     DOUBLE,
            room_type     TEXT,
            price         DOUBLE
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_reviews (
            review_id     BIGINT PRIMARY KEY,
            listing_id    BIGINT,
            review_date   DATE,
            reviewer_id   BIGINT,
            reviewer_name TEXT,
            comment       TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_pois (
            osm_id    BIGINT PRIMARY KEY,
            name      TEXT,
            category  TEXT,
            latitude  DOUBLE,
            longitude DOUBLE,
            tags      JSON
        )
    """)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_listings(conn: duckdb.DuckDBPyConnection, path: Path) -> int:
    """Read listings CSV/CSV.GZ, map columns, upsert into raw_listings."""
    df = pd.read_csv(path, usecols=lambda c: c in _LISTING_COLS, low_memory=False)

    # prefer the cleansed neighbourhood column when present
    if "neighbourhood_cleansed" in df.columns:
        df["neighbourhood"] = df["neighbourhood_cleansed"]
        df = df.drop(columns=["neighbourhood_cleansed"])
    if "neighbourhood" not in df.columns:
        df["neighbourhood"] = None

    df = df.rename(columns={"id": "listing_id"})

    # price arrives as "$1,234.56" in some snapshots
    df["price"] = pd.to_numeric(
        df["price"].astype(str).str.replace(r"[$,]", "", regex=True),
        errors="coerce",
    )

    df = df[["listing_id", "name", "neighbourhood", "latitude", "longitude", "room_type", "price"]]

    conn.execute("DELETE FROM raw_listings")
    conn.execute("INSERT INTO raw_listings SELECT * FROM df")
    count: int = conn.execute("SELECT COUNT(*) FROM raw_listings").fetchone()[0]
    log.info("raw_listings loaded: %d rows", count)
    return count


def _load_reviews(conn: duckdb.DuckDBPyConnection, path: Path) -> int:
    """Read reviews CSV/CSV.GZ, map columns, upsert into raw_reviews."""
    df = pd.read_csv(
        path,
        usecols=["id", "listing_id", "date", "reviewer_id", "reviewer_name", "comments"],
        low_memory=False,
    )
    df = df.rename(columns={"id": "review_id", "date": "review_date", "comments": "comment"})
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    df = df[["review_id", "listing_id", "review_date", "reviewer_id", "reviewer_name", "comment"]]

    conn.execute("DELETE FROM raw_reviews")
    conn.execute("INSERT INTO raw_reviews SELECT * FROM df")
    count: int = conn.execute("SELECT COUNT(*) FROM raw_reviews").fetchone()[0]
    log.info("raw_reviews loaded: %d rows", count)
    return count


def _load_pois(conn: duckdb.DuckDBPyConnection, path: Path) -> int:
    """Parse cached Overpass JSON and upsert into raw_pois."""
    elements = json.loads(path.read_text(encoding="utf-8")).get("elements", [])
    rows = []
    for el in elements:
        if el.get("type") != "node":
            continue
        tags = el.get("tags", {})
        rows.append({
            "osm_id":    el["id"],
            "name":      tags.get("name"),
            "category":  tags.get("amenity") or tags.get("tourism") or "unknown",
            "latitude":  el["lat"],
            "longitude": el["lon"],
            "tags":      json.dumps(tags),
        })

    if not rows:
        log.warning("No POI nodes found in %s", path)
        return 0

    df = pd.DataFrame(rows)
    conn.execute("DELETE FROM raw_pois")
    conn.execute("INSERT INTO raw_pois SELECT * FROM df")
    count: int = conn.execute("SELECT COUNT(*) FROM raw_pois").fetchone()[0]
    log.info("raw_pois loaded: %d rows", count)
    return count


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(refresh: bool = False) -> None:
    listings_path      = _AIRBNB_DIR / "listings.csv.gz"
    reviews_path       = _AIRBNB_DIR / "reviews.csv.gz"
    neighbourhoods_path = _AIRBNB_DIR / "neighbourhoods.geojson"
    pois_path          = _OSM_DIR / "edinburgh_pois.json"

    _download(INSIDE_AIRBNB_LISTINGS_URL, listings_path, refresh)
    _download(INSIDE_AIRBNB_REVIEWS_URL, reviews_path, refresh)
    _download(INSIDE_AIRBNB_NEIGHBOURHOODS_URL, neighbourhoods_path, refresh)
    _fetch_overpass(pois_path, refresh)

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(DB_PATH))
    try:
        _create_tables(conn)
        _load_listings(conn, listings_path)
        _load_reviews(conn, reviews_path)
        _load_pois(conn, pois_path)
    finally:
        conn.close()

    log.info("Ingest complete — database: %s", DB_PATH)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Ingest Edinburgh visitor data into DuckDB.")
    parser.add_argument("--refresh", action="store_true", help="Re-download even if cache exists.")
    args = parser.parse_args()
    run(refresh=args.refresh)


if __name__ == "__main__":
    main()
