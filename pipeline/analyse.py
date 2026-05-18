"""Phase 5 — embed reviews, cluster into 8 themes, label with TF-IDF."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler

from pipeline import metrics as metrics_module
from pipeline.config import DATA_RAW_DIR, DB_PATH

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_CLUSTERS   = 8
RANDOM_STATE = 42
BATCH_SIZE   = 64
MODEL_NAME   = "sentence-transformers/all-MiniLM-L6-v2"
TOP_N_TERMS  = 5
INTERIM_DIR  = DATA_RAW_DIR.parent / "interim"

# Generic / artifact tokens suppressed from TF-IDF labels.
# 'br' is an HTML <br> artifact that survived cleaning; the rest are
# Airbnb boilerplate that appear in every cluster and carry no signal.
_EXTRA_STOPS = frozenset({
    "br",                                                        # HTML artifact
    "edinburgh", "stay", "stayed", "place", "apartment", "flat",
    "host", "hosts", "great", "good", "nice", "lovely", "excellent",
    "highly", "recommend", "location", "area", "city", "visit", "visited",
    "just", "really", "very", "would", "also", "us", "like", "little",
    # Universal Airbnb-review words that drown out theme signal
    "clean", "perfect", "easy", "comfortable", "beautiful",
    "spacious", "close", "walk", "walking",
})


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _embed(texts: list[str], cache_path: Path) -> np.ndarray:
    """Return embeddings from cache, or compute → save → return."""
    if cache_path.exists():
        log.info("Loading cached embeddings from %s …", cache_path.name)
        return np.load(cache_path)

    log.info("Embedding %d texts with %s (batch_size=%d) …", len(texts), MODEL_NAME, BATCH_SIZE)
    model = SentenceTransformer(MODEL_NAME)
    embeddings: np.ndarray = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    log.info("Embeddings cached → %s", cache_path)
    return embeddings


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def _cluster(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """KMeans → (cluster_id per row, distance matrix shape (n, k))."""
    log.info("KMeans k=%d random_state=%d …", N_CLUSTERS, RANDOM_STATE)
    km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init="auto")
    labels    = km.fit_predict(embeddings)
    distances = km.transform(embeddings)   # shape (n_reviews, n_clusters)
    return labels, distances


# ---------------------------------------------------------------------------
# Theme labelling — comparative document-frequency approach
# ---------------------------------------------------------------------------

def _build_doc_freq_matrix(texts: list[str]) -> tuple:
    """
    Fit a binary CountVectorizer on all texts.
    Returns (binary_matrix, feature_names) where matrix[i,j]=1 if doc i contains term j.
    Binary doc-frequency is more robust than TF-IDF mean for cluster characterisation.
    """
    vec = CountVectorizer(
        max_features=5000,
        stop_words="english",
        min_df=5,
        ngram_range=(1, 1),
        binary=True,
    )
    matrix        = vec.fit_transform(texts)
    feature_names = vec.get_feature_names_out()
    return matrix, feature_names


def _label_cluster(
    doc_freq_matrix,
    feature_names: np.ndarray,
    cluster_mask: np.ndarray,
    n_docs: int,
) -> str:
    """
    Score = cluster_prevalence × (cluster_prevalence / global_prevalence).
    This rewards terms that are BOTH frequent in the cluster AND overrepresented
    vs. the corpus, avoiding the trap of picking rare-but-unique noise terms.
    A minimum cluster prevalence gate (≥3 % of cluster docs) is also applied.
    """
    cluster_size = int(cluster_mask.sum())
    if cluster_size == 0:
        return "unlabelled"

    cluster_df = doc_freq_matrix[cluster_mask].sum(axis=0).A1 / cluster_size
    global_df  = doc_freq_matrix.sum(axis=0).A1 / n_docs

    min_prevalence = 0.03   # term must appear in ≥3 % of cluster docs
    score = np.where(
        cluster_df >= min_prevalence,
        cluster_df * (cluster_df / (global_df + 1e-9)),
        0.0,
    )

    candidates = [
        (feature_names[i], score[i])
        for i in range(len(feature_names))
        if score[i] > 0 and feature_names[i] not in _EXTRA_STOPS
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)
    terms = [t for t, _ in candidates[:TOP_N_TERMS]]
    return " · ".join(terms) if terms else "unlabelled"


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _create_themes_table(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("DROP TABLE IF EXISTS review_themes")
    conn.execute("""
        CREATE TABLE review_themes (
            review_id   BIGINT,
            theme_id    INT,
            theme_label TEXT,
            weight      DOUBLE
        )
    """)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(sample: int | None = None) -> dict:
    # ── Load reviews ──────────────────────────────────────────────────────
    conn = duckdb.connect(str(DB_PATH))
    try:
        reviews_df = conn.execute(
            "SELECT review_id, comment_clean FROM clean_reviews ORDER BY review_id"
        ).df()
    finally:
        conn.close()

    is_sample = sample is not None and sample < len(reviews_df)
    if is_sample:
        reviews_df = reviews_df.sample(n=sample, random_state=RANDOM_STATE).reset_index(drop=True)
        log.info("Sample mode: %d of %d total reviews", len(reviews_df), sample)

    texts      = reviews_df["comment_clean"].fillna("").tolist()
    review_ids = reviews_df["review_id"].to_numpy()

    # ── Embed ─────────────────────────────────────────────────────────────
    suffix     = f"_sample_{sample}" if is_sample else ""
    cache_path = INTERIM_DIR / f"review_embeddings{suffix}.npy"
    embeddings = _embed(texts, cache_path)

    # ── Cluster ───────────────────────────────────────────────────────────
    labels, distances = _cluster(embeddings)

    # ── Label each cluster ────────────────────────────────────────────────
    log.info("Building document-frequency matrix for cluster labelling …")
    doc_freq_matrix, feature_names = _build_doc_freq_matrix(texts)
    n_docs = len(texts)

    log.info("Computing cluster labels (%d clusters) …", N_CLUSTERS)
    theme_labels: dict[int, str]  = {}
    cluster_sizes: dict[int, int] = {}

    for cid in range(N_CLUSTERS):
        mask               = labels == cid
        count              = int(mask.sum())
        cluster_sizes[cid] = count
        label              = _label_cluster(doc_freq_matrix, feature_names, mask, n_docs)
        theme_labels[cid]  = label
        log.info("  Cluster %d (%5d reviews): %s", cid, count, label)

    # ── Weights (normalised distance to own centroid, 0 = closest) ────────
    own_distances = distances[np.arange(len(labels)), labels]
    weights = MinMaxScaler().fit_transform(own_distances.reshape(-1, 1)).ravel()

    # ── Build output DataFrame ────────────────────────────────────────────
    themes_df = pd.DataFrame({
        "review_id":   review_ids,
        "theme_id":    labels.astype(int),
        "theme_label": [theme_labels[c] for c in labels],
        "weight":      np.round(weights, 6),
    })

    # ── Write to DB ───────────────────────────────────────────────────────
    conn = duckdb.connect(str(DB_PATH))
    try:
        _create_themes_table(conn)
        conn.execute("INSERT INTO review_themes SELECT * FROM themes_df")
        log.info("review_themes: %d rows written", len(themes_df))
    finally:
        conn.close()

    # ── Metrics ───────────────────────────────────────────────────────────
    m: dict = {
        "theme_count":         N_CLUSTERS,
        "theme_labels":        theme_labels,
        "theme_cluster_sizes": cluster_sizes,
    }
    if not is_sample:
        metrics_module.write({
            "theme_count":     N_CLUSTERS,
            "avg_cluster_size": round(sum(cluster_sizes.values()) / N_CLUSTERS, 1),
            "theme_labels":    list(theme_labels.values()),
        })

    return m


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Cluster reviews into themes.")
    parser.add_argument(
        "--sample", type=int, default=None, metavar="N",
        help="Use N random reviews instead of the full dataset",
    )
    args = parser.parse_args()

    m = run(sample=args.sample)
    log.info("Analyse complete.\n%s", json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
