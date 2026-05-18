"""
Happy-path tests for pipeline.analyse labelling logic.

Avoids loading sentence-transformers or touching DuckDB — all assertions
are on the pure-Python/_numpy functions that do cluster characterisation.
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from pipeline.analyse import _label_cluster, _EXTRA_STOPS


@pytest.fixture
def tiny_corpus():
    """
    6 docs × 5 terms.  Two clear clusters:
      A (docs 0-2): heavy on 'castle' and 'tour'
      B (docs 3-5): heavy on 'kitchen' and 'wifi'
    'great' appears in every doc → global noise, should never be a top label.
    """
    # columns: castle, tour, kitchen, wifi, great
    dense = np.array([
        [1, 1, 0, 0, 1],
        [1, 1, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1],
    ], dtype=float)
    feature_names = np.array(["castle", "tour", "kitchen", "wifi", "great"])
    return csr_matrix(dense), feature_names


def test_label_cluster_picks_cluster_specific_terms(tiny_corpus):
    """
    Cluster A's label should contain 'castle' and 'tour' (overrepresented vs. corpus),
    not 'kitchen' or 'wifi' (absent in cluster A).
    """
    matrix, feature_names = tiny_corpus
    mask_a = np.array([True, True, True, False, False, False])

    label = _label_cluster(matrix, feature_names, mask_a, n_docs=6)

    assert "castle" in label, f"Expected 'castle' in cluster-A label, got: {label}"
    assert "tour"   in label, f"Expected 'tour'   in cluster-A label, got: {label}"
    assert "kitchen" not in label
    assert "wifi"    not in label


def test_label_cluster_suppresses_ubiquitous_terms(tiny_corpus):
    """
    'great' appears in every document so its cluster_df ≈ global_df → ratio ≈ 1.
    Its score (cluster_df × ratio) will be beaten by terms with higher overrepresentation.
    It must not appear as the top label for either cluster.
    """
    matrix, feature_names = tiny_corpus
    mask_b = np.array([False, False, False, True, True, True])

    label = _label_cluster(matrix, feature_names, mask_b, n_docs=6)

    # 'great' is not in _EXTRA_STOPS in this fixture, so the scoring must handle it
    terms = label.split(" · ")
    assert terms[0] != "great", f"'great' should not be the top term; got: {label}"
    assert "kitchen" in label or "wifi" in label, f"Cluster-B label missing amenity terms: {label}"


def test_extra_stops_excludes_known_noise():
    """Spot-check that key noise tokens are in the stop set."""
    for token in ("br", "edinburgh", "clean", "perfect", "walk"):
        assert token in _EXTRA_STOPS, f"Expected '{token}' in _EXTRA_STOPS"


def test_label_cluster_empty_cluster_returns_unlabelled(tiny_corpus):
    matrix, feature_names = tiny_corpus
    empty_mask = np.zeros(6, dtype=bool)
    assert _label_cluster(matrix, feature_names, empty_mask, n_docs=6) == "unlabelled"
