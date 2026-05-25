"""
Happy-path tests for pipeline.geocode.

3 POIs + 4 reviews demonstrate exact (word-boundary) vs NER+fuzzy behaviour:
  review 1 — "Old Town"           exact hit  (\\bold town\\b matches)
  review 2 — "Grassmarket"        exact hit
  review 3 — "Waverley Stations"  exact MISS (\\bwaverley station\\b requires a
                                  word boundary after 'station'; 'stations' has
                                  no boundary there → blocked, as intended)
  review 4 — "Grass Market"       exact MISS (space vs no-space); mock NER tags
                                  it as LOC; rapidfuzz scores "grass market" vs
                                  "grassmarket" at ~96 → fuzzy hit.

Result: exact finds 2 links, NER+fuzzy finds 3 → proved uplift.
"""

import pandas as pd
import pytest

from pipeline.geocode import _exact_match, _ner_fuzzy_match


# ---------------------------------------------------------------------------
# Minimal mock NLP — avoids spaCy model loading and brittle entity tagging
# ---------------------------------------------------------------------------

class _Ent:
    def __init__(self, text, label):
        self.text   = text
        self.label_ = label

class _Doc:
    def __init__(self, ents):
        self.ents = ents

class _MockNLP:
    """Returns 'Grass Market' as LOC for review 4; empty docs for all others."""
    def pipe(self, texts, **kwargs):
        for text in texts:
            if "Grass Market" in text:
                yield _Doc([_Ent("Grass Market", "LOC")])
            else:
                yield _Doc([])


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fixture_data():
    # Listing 101 sits at 55.9530, -3.1880 — all three POIs are within 1 km
    pois = pd.DataFrame([
        {"poi_id": 1, "canonical_name": "old town",
         "latitude": 55.9490, "longitude": -3.1920},
        {"poi_id": 2, "canonical_name": "grassmarket",
         "latitude": 55.9460, "longitude": -3.1940},
        {"poi_id": 3, "canonical_name": "waverley station",
         "latitude": 55.9520, "longitude": -3.1890},
    ])
    reviews = pd.DataFrame([
        {"review_id": 1, "listing_id": 101,
         "comment_clean": "We stayed in the Old Town, fantastic location!"},
        {"review_id": 2, "listing_id": 101,
         "comment_clean": "Visited the Grassmarket area, brilliant food stalls."},
        {"review_id": 3, "listing_id": 101,
         "comment_clean": "Caught a train from Waverley Stations to get home."},
        {"review_id": 4, "listing_id": 101,
         "comment_clean": "The Grass Market was a highlight of our trip."},
    ])
    listing_loc = {101: (55.9530, -3.1880)}
    name_to_id  = dict(zip(pois["canonical_name"], pois["poi_id"]))
    return pois, reviews, listing_loc, name_to_id


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_exact_match_word_boundaries(fixture_data):
    """
    Word-boundary pass finds 2 of 4 reviews.
    'waverley station' does NOT match 'waverley stations' — \\b blocks the plural.
    'grassmarket' does NOT match 'grass market' — space breaks the token.
    """
    _, reviews, _, name_to_id = fixture_data
    links = _exact_match(reviews, name_to_id)
    pairs = {(d["review_id"], d["poi_id"]) for d in links}

    assert len(links) == 2, f"Expected 2 exact links, got {len(links)}: {links}"
    assert (1, 1) in pairs, "Old Town should be an exact match"
    assert (2, 2) in pairs, "Grassmarket should be an exact match"
    assert (3, 3) not in pairs, r"\bwaverley station\b must NOT match 'waverley stations'"
    assert (4, 2) not in pairs, "'grassmarket' must NOT match 'grass market' (space)"


def test_fuzzy_beats_exact(fixture_data):
    """Mock NER tags 'Grass Market' as LOC; rapidfuzz scores it vs 'grassmarket' at ~96 → fuzzy hit."""
    pois, reviews, listing_loc, name_to_id = fixture_data
    nlp = _MockNLP()

    exact_links  = _exact_match(reviews, name_to_id)
    exact_pairs  = {(d["review_id"], d["poi_id"]) for d in exact_links}

    ner_links = _ner_fuzzy_match(
        reviews, pois, listing_loc, nlp, existing_pairs=exact_pairs
    )
    ner_pairs = {(d["review_id"], d["poi_id"]) for d in ner_links}

    total = len(exact_links) + len(ner_links)
    assert total > len(exact_links), (
        f"NER+fuzzy ({total}) must exceed exact ({len(exact_links)})"
    )
    assert (4, 2) in ner_pairs, (
        f"Expected review 4 → poi 2 ('Grass Market' → 'grassmarket'). "
        f"NER links: {ner_links}"
    )
