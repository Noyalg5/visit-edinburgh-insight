"""Happy-path tests for pipeline.ingest loaders."""
import json
from pathlib import Path

import duckdb
import pytest

from pipeline.ingest import _create_tables, _load_listings, _load_pois, _load_reviews

FIXTURES = Path(__file__).parent / "fixtures"


def test_ingest_loaders(tmp_path: Path) -> None:
    """All three loaders correctly populate DuckDB tables from fixture data."""
    conn = duckdb.connect(":memory:")
    _create_tables(conn)

    # --- listings ---
    n_listings = _load_listings(conn, FIXTURES / "listings.csv")
    assert n_listings == 3
    price = conn.execute(
        "SELECT price FROM raw_listings WHERE listing_id = 1"
    ).fetchone()[0]
    assert price == 75.0
    nbhd = conn.execute(
        "SELECT neighbourhood FROM raw_listings WHERE listing_id = 2"
    ).fetchone()[0]
    assert nbhd == "New Town"

    # --- reviews ---
    n_reviews = _load_reviews(conn, FIXTURES / "reviews.csv")
    assert n_reviews == 3
    comment = conn.execute(
        "SELECT comment FROM raw_reviews WHERE review_id = 101"
    ).fetchone()[0]
    assert comment == "Great place to stay in Edinburgh!"

    # --- pois (built inline, no network call) ---
    poi_json = {
        "elements": [
            {
                "type": "node", "id": 1001, "lat": 55.9490, "lon": -3.1910,
                "tags": {"amenity": "restaurant", "name": "The Witchery"},
            },
            {
                "type": "node", "id": 1002, "lat": 55.9500, "lon": -3.1950,
                "tags": {"tourism": "museum", "name": "National Museum"},
            },
        ]
    }
    poi_path = tmp_path / "pois.json"
    poi_path.write_text(json.dumps(poi_json))

    n_pois = _load_pois(conn, poi_path)
    assert n_pois == 2
    category = conn.execute(
        "SELECT category FROM raw_pois WHERE osm_id = 1001"
    ).fetchone()[0]
    assert category == "restaurant"

    conn.close()
