"""
Rule-based insight cards for the Recommendations tab.

Each rule is a function that accepts a themes_df DataFrame
(columns: theme_label, review_date, neighbourhood) representing the
currently-filtered dataset, and returns a card dict or None if the
rule does not fire.

Card schema:
  title            str   — headline shown in the card header
  suggestion       str   — actionable recommendation
  evidence_text    str   — one-liner supporting the number
  evidence_numbers int   — key count / volume that triggered the rule
  neighbourhood    str | None
"""
from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_theme(themes_df: pd.DataFrame, keywords: list[str]) -> str | None:
    """Return first theme_label that contains any of the given keywords."""
    for label in themes_df["theme_label"].unique():
        if any(kw in label for kw in keywords):
            return label
    return None


def _hood_share(themes_df: pd.DataFrame, theme: str) -> pd.Series:
    """Per-neighbourhood fraction of reviews matching theme."""
    sub = themes_df[themes_df["neighbourhood"].notna()].copy()
    sub["match"] = (sub["theme_label"] == theme)
    return sub.groupby("neighbourhood")["match"].mean()


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

def rule_theme_volume_spike(themes_df: pd.DataFrame) -> dict | None:
    """
    Compare relative theme share in the second half vs. first half of the
    filtered date range. Fire if any theme grew ≥ 25 %.
    """
    if "review_date" not in themes_df.columns or len(themes_df) < 100:
        return None

    mid = themes_df["review_date"].median()
    if pd.isna(mid):
        return None

    first  = themes_df[themes_df["review_date"] <= mid]
    second = themes_df[themes_df["review_date"] > mid]
    if len(first) < 20 or len(second) < 20:
        return None

    fh = first["theme_label"].value_counts(normalize=True)
    sh = second["theme_label"].value_counts(normalize=True)
    growth = {
        t: (sh.get(t, 0) - fh[t]) / fh[t]
        for t in fh.index
        if fh[t] > 0 and t in sh.index
    }
    if not growth:
        return None

    top = max(growth, key=growth.get)
    if growth[top] < 0.25:
        return None

    return {
        "title":            "Rising Theme Detected",
        "suggestion":       (
            f"'{top}' grew {growth[top]*100:.0f}% in relative share in the "
            f"second half of the selected period. Consider marketing content "
            f"that directly addresses this visitor priority."
        ),
        "evidence_text":    f"Share: {fh.get(top,0)*100:.1f}% → {sh.get(top,0)*100:.1f}%",
        "evidence_numbers": int(second[second["theme_label"] == top].shape[0]),
        "neighbourhood":    None,
    }


def rule_self_catering_neighbourhood(themes_df: pd.DataFrame) -> dict | None:
    """
    Neighbourhood where self-catering / amenities theme is ≥ 1.5× city average.
    Signals a kitchen-equipped marketing angle.
    """
    theme = _find_theme(themes_df, ["kitchen", "bathroom"])
    if theme is None or "neighbourhood" not in themes_df.columns:
        return None

    city_share = (themes_df["theme_label"] == theme).mean()
    if city_share == 0:
        return None

    hood = _hood_share(themes_df, theme)
    if hood.empty:
        return None

    top_hood, top_share = hood.idxmax(), hood.max()
    if top_share < city_share * 1.5:
        return None

    count = int(themes_df[
        (themes_df["neighbourhood"] == top_hood) & (themes_df["theme_label"] == theme)
    ].shape[0])

    return {
        "title":            f"Self-Catering Demand Signal: {top_hood}",
        "suggestion":       (
            f"{top_hood} over-indexes on self-catering amenities "
            f"({top_share*100:.0f}% vs {city_share*100:.0f}% city average). "
            f"Highlight kitchen-equipped properties in marketing for this area."
        ),
        "evidence_text":    f"'{theme}' is {top_share/city_share:.1f}× city average in {top_hood}",
        "evidence_numbers": count,
        "neighbourhood":    top_hood,
    }


def rule_sightseeing_hotspot(themes_df: pd.DataFrame) -> dict | None:
    """
    Neighbourhood where the Royal Mile / sightseeing theme accounts for
    ≥ 20 % of reviews — coordinate on visitor flow and off-peak promotions.
    """
    theme = _find_theme(themes_df, ["royal", "castle", "mile", "town"])
    if theme is None or "neighbourhood" not in themes_df.columns:
        return None

    city_share = (themes_df["theme_label"] == theme).mean()
    hood = _hood_share(themes_df, theme)
    if hood.empty:
        return None

    top_hood, top_share = hood.idxmax(), hood.max()
    if top_share < 0.20:
        return None

    count = int(themes_df[
        (themes_df["neighbourhood"] == top_hood) & (themes_df["theme_label"] == theme)
    ].shape[0])

    return {
        "title":            f"Royal Mile / Sightseeing Hotspot: {top_hood}",
        "suggestion":       (
            f"{top_hood} has a high concentration of Old Town and Royal Mile "
            f"mentions ({top_share*100:.0f}% of its reviews). Coordinate with "
            f"attractions on visitor-flow management and off-peak promotions."
        ),
        "evidence_text":    (
            f"'{theme}' at {top_share*100:.0f}% in {top_hood} "
            f"vs {city_share*100:.0f}% city-wide"
        ),
        "evidence_numbers": count,
        "neighbourhood":    top_hood,
    }


def rule_host_praise_hotspot(themes_df: pd.DataFrame) -> dict | None:
    """
    Neighbourhood with ≥ 1.3× city-average host-praise share.
    Competitive differentiator worth showcasing in destination marketing.
    """
    theme = _find_theme(themes_df, ["friendly", "responsive", "welcoming"])
    if theme is None or "neighbourhood" not in themes_df.columns:
        return None

    city_share = (themes_df["theme_label"] == theme).mean()
    hood = _hood_share(themes_df, theme)
    if hood.empty:
        return None

    top_hood, top_share = hood.idxmax(), hood.max()
    if top_share < city_share * 1.3:
        return None

    count = int(themes_df[
        (themes_df["neighbourhood"] == top_hood) & (themes_df["theme_label"] == theme)
    ].shape[0])

    return {
        "title":            f"Host Quality Signal: {top_hood}",
        "suggestion":       (
            f"Guests in {top_hood} disproportionately praise host responsiveness "
            f"and warmth. Feature host profiles and response-rate badges for "
            f"this area in destination marketing."
        ),
        "evidence_text":    (
            f"Host-praise theme {top_share/city_share:.1f}× city average "
            f"({top_share*100:.0f}% vs {city_share*100:.0f}%)"
        ),
        "evidence_numbers": count,
        "neighbourhood":    top_hood,
    }


def rule_repeat_visitor_signal(themes_df: pd.DataFrame) -> dict | None:
    """
    Neighbourhood with ≥ 1.3× city-average share of home / return-visit reviews.
    Strong loyalty segment for repeat-stay campaigns.
    """
    theme = _find_theme(themes_df, ["home", "wonderful", "loved", "enjoyed"])
    if theme is None or "neighbourhood" not in themes_df.columns:
        return None

    city_share = (themes_df["theme_label"] == theme).mean()
    hood = _hood_share(themes_df, theme)
    if hood.empty:
        return None

    top_hood, top_share = hood.idxmax(), hood.max()
    if top_share < city_share * 1.3:
        return None

    count = int(themes_df[
        (themes_df["neighbourhood"] == top_hood) & (themes_df["theme_label"] == theme)
    ].shape[0])

    return {
        "title":            f"Loyal Guest Segment: {top_hood}",
        "suggestion":       (
            f"{top_hood} shows a strong home-away-from-home and return-visit "
            f"sentiment. Target with loyalty incentives and early-return booking "
            f"prompts."
        ),
        "evidence_text":    (
            f"Return-visit theme {top_share/city_share:.1f}× city average in "
            f"{top_hood} ({top_share*100:.0f}% vs {city_share*100:.0f}%)"
        ),
        "evidence_numbers": count,
        "neighbourhood":    top_hood,
    }


def rule_quiet_location_demand(themes_df: pd.DataFrame) -> dict | None:
    """
    City-wide quiet-location theme share ≥ 8 % signals an under-served segment.
    """
    theme = _find_theme(themes_df, ["quiet"])
    if theme is None:
        return None

    city_share = (themes_df["theme_label"] == theme).mean()
    if city_share < 0.08:
        return None

    counts = themes_df["theme_label"].value_counts()
    rank   = (list(counts.index).index(theme) + 1) if theme in counts.index else "?"
    sfx    = {1: "st", 2: "nd", 3: "rd"}.get(rank, "th") if isinstance(rank, int) else ""
    count  = int((themes_df["theme_label"] == theme).sum())

    return {
        "title":            "Quiet-Location Demand Signal",
        "suggestion":       (
            f"{city_share*100:.0f}% of reviews highlight quiet location as a key "
            f"theme (the {rank}{sfx} most common). Proactively tag and surface "
            f"'quiet street / away from nightlife' properties."
        ),
        "evidence_text":    f"Theme '{theme}' at {city_share*100:.0f}% of filtered reviews",
        "evidence_numbers": count,
        "neighbourhood":    None,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

_RULES = [
    rule_theme_volume_spike,
    rule_self_catering_neighbourhood,
    rule_sightseeing_hotspot,
    rule_host_praise_hotspot,
    rule_repeat_visitor_signal,
    rule_quiet_location_demand,
]


def get_recommendations(themes_df: pd.DataFrame) -> list[dict]:
    """Run all rules; swallow individual failures so one bad rule can't crash the tab."""
    cards = []
    for rule in _RULES:
        try:
            card = rule(themes_df)
            if card is not None:
                cards.append(card)
        except Exception:
            pass
    return cards
