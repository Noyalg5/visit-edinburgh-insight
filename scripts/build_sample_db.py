"""
Build db/visit_sample.duckdb for Streamlit Cloud deployment.

The full DB (662 MB) is too large to commit. This script creates a slim
sample containing all POIs + 30,000 randomly sampled reviews plus their
matching review_poi_links and review_themes rows, so the deployed dashboard
is fully functional with a representative slice of the data.

Usage:
    python scripts/build_sample_db.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FULL_DB    = PROJECT_ROOT / "db" / "visit.duckdb"
SAMPLE_DB  = PROJECT_ROOT / "db" / "visit_sample.duckdb"
SAMPLE_N   = 30_000


def main() -> None:
    if not FULL_DB.exists():
        print(f"Full DB not found at {FULL_DB}. Run the pipeline first.")
        sys.exit(1)

    SAMPLE_DB.unlink(missing_ok=True)

    con = duckdb.connect(str(SAMPLE_DB))
    con.execute(f"ATTACH '{FULL_DB}' AS src (READ_ONLY)")

    try:
        # ── raw_listings (needed for listing lat/lon lookups) ─────────────
        con.execute("""
            CREATE TABLE raw_listings AS
            SELECT * FROM src.raw_listings
        """)

        # ── clean_pois (all 2,270 rows — small) ──────────────────────────
        con.execute("""
            CREATE TABLE clean_pois AS
            SELECT * FROM src.clean_pois
        """)

        # ── clean_reviews (random 30 K using reservoir-style ORDER BY) ────
        con.execute(f"""
            CREATE TABLE clean_reviews AS
            SELECT * FROM src.clean_reviews
            USING SAMPLE {SAMPLE_N} ROWS (reservoir, 42)
        """)

        # ── review_poi_links (only for sampled reviews) ───────────────────
        con.execute("""
            CREATE TABLE review_poi_links AS
            SELECT l.* FROM src.review_poi_links l
            WHERE l.review_id IN (SELECT review_id FROM clean_reviews)
        """)

        # ── review_themes (only for sampled reviews) ──────────────────────
        con.execute("""
            CREATE TABLE review_themes AS
            SELECT t.* FROM src.review_themes t
            WHERE t.review_id IN (SELECT review_id FROM clean_reviews)
        """)

        # ── summary ───────────────────────────────────────────────────────
        n_rev   = con.execute("SELECT COUNT(*) FROM clean_reviews").fetchone()[0]
        n_poi   = con.execute("SELECT COUNT(*) FROM clean_pois").fetchone()[0]
        n_links = con.execute("SELECT COUNT(*) FROM review_poi_links").fetchone()[0]
        n_theme = con.execute("SELECT COUNT(*) FROM review_themes").fetchone()[0]
        size_mb = SAMPLE_DB.stat().st_size / 1_048_576

        print(f"Sample DB written to {SAMPLE_DB}")
        print(f"  clean_reviews    : {n_rev:,}")
        print(f"  clean_pois       : {n_poi:,}")
        print(f"  review_poi_links : {n_links:,}")
        print(f"  review_themes    : {n_theme:,}")
        print(f"  File size        : {size_mb:.1f} MB")

    finally:
        con.close()


if __name__ == "__main__":
    main()
