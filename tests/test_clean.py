"""Happy-path tests for pipeline.clean POI normalisation and deduplication."""

import pandas as pd

from pipeline.clean import _dedupe_pois, _normalise_name


def test_normalise_name() -> None:
    assert _normalise_name("St Andrews Bar")  == "street andrews bar"
    assert _normalise_name("St. Andrews Bar") == "street andrews bar"
    assert _normalise_name("Rd End Cafe")     == "road end cafe"
    assert _normalise_name("5th Ave Bistro")  == "5th avenue bistro"
    assert _normalise_name(None)              == ""


def test_dedupe_pois_five_row_fixture() -> None:
    """
    5-row fixture:
      osm 1 — "St Andrews Bar"   55.9500, -3.1900  ← cluster seed
      osm 2 — "St. Andrews Bar"  55.9501, -3.1901  ← proximity dup  (~13 m)
      osm 3 — "st andrews bar"   55.9500, -3.1900  ← exact name dup (0 m)
      osm 4 — "The Castle Pub"   55.9400, -3.2000  ← unique
      osm 5 — "Cafe Noir"        55.9600, -3.1500  ← unique

    All three of osm 1-3 normalise to "street andrews bar" and sit within 50 m
    of each other → collapse to 1 clean_pois row.
    Expected output: 3 rows total.
    """
    raw = pd.DataFrame([
        {"osm_id": 1, "name": "St Andrews Bar",  "category": "bar",  "latitude": 55.9500, "longitude": -3.1900},
        {"osm_id": 2, "name": "St. Andrews Bar", "category": "bar",  "latitude": 55.9501, "longitude": -3.1901},
        {"osm_id": 3, "name": "st andrews bar",  "category": "bar",  "latitude": 55.9500, "longitude": -3.1900},
        {"osm_id": 4, "name": "The Castle Pub",  "category": "pub",  "latitude": 55.9400, "longitude": -3.2000},
        {"osm_id": 5, "name": "Cafe Noir",        "category": "cafe", "latitude": 55.9600, "longitude": -3.1500},
    ])

    result = _dedupe_pois(raw)

    assert len(result) == 3, f"Expected 3 rows, got {len(result)}: {result['canonical_name'].tolist()}"

    merged = result[result["canonical_name"] == "street andrews bar"].iloc[0]
    assert sorted(merged["source_osm_ids"]) == [1, 2, 3]

    assert set(result["canonical_name"]) == {"street andrews bar", "the castle pub", "cafe noir"}
