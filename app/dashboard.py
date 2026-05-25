"""app/dashboard.py — VisitEdinburgh Insight Streamlit dashboard.

Three tabs: Overview (KPIs + map + top POIs), Themes (stacked area chart +
neighbourhood bar + sample reviews), Recommendations (rule-based cards).

All heavy reads go through @st.cache_data so re-filtering is snappy.
Run with:  streamlit run app/dashboard.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

# Streamlit adds the script's directory (app/) to sys.path, not the project root.
# This insert makes `from app.rules` and `from pipeline.config` resolve correctly
# whether this file is run via `streamlit run app/dashboard.py` OR imported from
# the project root with `import app.dashboard`.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

from app.rules import get_recommendations
from pipeline.config import DB_PATH

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CENTRE = {"lat": 55.9533, "lon": -3.1883}

# Okabe-Ito colorblind-safe palette (8 distinct hues + grey fallback)
_CAT_COLORS: dict[str, list[int]] = {
    "restaurant": [230, 159,   0, 220],   # orange
    "cafe":       [ 86, 180, 233, 220],   # sky blue
    "bar":        [213,  94,   0, 220],   # vermillion
    "pub":        [204, 121, 167, 220],   # reddish purple
    "hotel":      [  0, 114, 178, 220],   # blue
    "museum":     [  0, 158, 115, 220],   # bluish green
    "attraction": [230,  97,   0, 220],   # deep orange
    "gallery":    [120, 120, 120, 220],   # grey
    "viewpoint":  [100, 176,  55, 220],   # green
    "fast_food":  [163,  75, 139, 220],   # purple
}
_DEFAULT_COLOR = [100, 100, 100, 180]


# ---------------------------------------------------------------------------
# DuckDB helpers
# ---------------------------------------------------------------------------

def _conn() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(DB_PATH), read_only=True)


def _in(col: str, values: tuple, params: list) -> str:
    """Append IN-clause params and return the SQL fragment (or empty string)."""
    if not values:
        return ""
    placeholders = ", ".join(["?" for _ in values])
    params.extend(list(values))
    return f" AND {col} IN ({placeholders})"


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def load_filter_options() -> dict:
    con = _conn()
    try:
        d_min, d_max = con.execute(
            "SELECT MIN(review_date), MAX(review_date) FROM clean_reviews"
        ).fetchone()
        hoods = [r[0] for r in con.execute(
            "SELECT DISTINCT neighbourhood FROM clean_reviews "
            "WHERE neighbourhood IS NOT NULL ORDER BY neighbourhood"
        ).fetchall()]
        cats = [r[0] for r in con.execute(
            "SELECT DISTINCT category FROM clean_pois "
            "WHERE category IS NOT NULL ORDER BY category"
        ).fetchall()]
    finally:
        con.close()
    return {"date_min": d_min, "date_max": d_max, "hoods": hoods, "cats": cats}


@st.cache_data(ttl=600)
def load_kpis(
    start: date, end: date, hoods: tuple, cats: tuple,
) -> dict:
    con = _conn()
    try:
        p: list = [start, end]
        q = "SELECT COUNT(*) FROM clean_reviews WHERE review_date BETWEEN ? AND ?"
        q += _in("neighbourhood", hoods, p)
        total_reviews = con.execute(q, p).fetchone()[0]

        p2: list = []
        q2 = "SELECT COUNT(*) FROM clean_pois WHERE 1=1"
        q2 += _in("category", cats, p2)
        total_pois = con.execute(q2, p2).fetchone()[0]

        months = max(1, (end.year - start.year) * 12 + (end.month - start.month) + 1)
    finally:
        con.close()

    return {
        "total_reviews": total_reviews,
        "total_pois":    total_pois,
        "date_range":    f"{start.strftime('%b %Y')} – {end.strftime('%b %Y')}",
        "avg_per_month": round(total_reviews / months, 1),
    }


@st.cache_data(ttl=600)
def load_poi_layer(
    start: date, end: date, hoods: tuple, cats: tuple,
) -> pd.DataFrame:
    con = _conn()
    try:
        # Subquery: reviews matching date + neighbourhood filter
        inner_p: list = [start, end]
        inner_q = (
            "SELECT l.poi_id, l.review_id "
            "FROM review_poi_links l "
            "JOIN clean_reviews r ON l.review_id = r.review_id "
            "WHERE r.review_date BETWEEN ? AND ?"
        )
        inner_q += _in("r.neighbourhood", hoods, inner_p)

        outer_p = inner_p.copy()
        outer_where = ""
        if cats:
            placeholders = ", ".join(["?" for _ in cats])
            outer_p.extend(list(cats))
            outer_where = f"WHERE p.category IN ({placeholders})"

        q = f"""
            SELECT p.poi_id, p.canonical_name, p.category,
                   p.latitude, p.longitude,
                   COUNT(DISTINCT f.review_id) AS mention_count
            FROM clean_pois p
            LEFT JOIN ({inner_q}) f ON p.poi_id = f.poi_id
            {outer_where}
            GROUP BY 1, 2, 3, 4, 5
            ORDER BY mention_count DESC
        """
        df = con.execute(q, outer_p).df()
    finally:
        con.close()

    # log10 compresses 50K:3 (≈16 000×) down to 4.7:0.5 (≈9×), then
    # radius_min/max_pixels in the layer clamps visual size to 3–30 px.
    df["log_radius"] = np.log10(df["mention_count"].clip(lower=1) + 1)
    colors = df["category"].apply(lambda c: _CAT_COLORS.get(c, _DEFAULT_COLOR))
    df[["r", "g", "b", "a"]] = pd.DataFrame(colors.tolist(), index=df.index)
    return df


@st.cache_data(ttl=600)
def load_theme_monthly(start: date, end: date, hoods: tuple) -> pd.DataFrame:
    con = _conn()
    try:
        p: list = [start, end]
        q = (
            "SELECT DATE_TRUNC('month', r.review_date) AS month, "
            "       t.theme_label, COUNT(*) AS volume "
            "FROM review_themes t "
            "JOIN clean_reviews r ON t.review_id = r.review_id "
            "WHERE r.review_date BETWEEN ? AND ?"
        )
        q += _in("r.neighbourhood", hoods, p)
        q += " GROUP BY 1, 2 ORDER BY 1, 2"
        df = con.execute(q, p).df()
    finally:
        con.close()
    return df


@st.cache_data(ttl=600)
def load_theme_by_neighbourhood(start: date, end: date, hoods: tuple) -> pd.DataFrame:
    con = _conn()
    try:
        p: list = [start, end]
        q = (
            "SELECT t.theme_label, r.neighbourhood, COUNT(*) AS volume "
            "FROM review_themes t "
            "JOIN clean_reviews r ON t.review_id = r.review_id "
            "WHERE r.review_date BETWEEN ? AND ? "
            "  AND r.neighbourhood IS NOT NULL"
        )
        q += _in("r.neighbourhood", hoods, p)
        q += " GROUP BY 1, 2 ORDER BY 3 DESC"
        df = con.execute(q, p).df()
    finally:
        con.close()
    return df


@st.cache_data(ttl=600)
def load_sample_reviews(
    theme_label: str, start: date, end: date, hoods: tuple,
) -> pd.DataFrame:
    con = _conn()
    try:
        p: list = [theme_label, start, end]
        q = (
            "SELECT r.listing_id, LEFT(r.comment_clean, 200) AS snippet "
            "FROM review_themes t "
            "JOIN clean_reviews r ON t.review_id = r.review_id "
            "WHERE t.theme_label = ? "
            "  AND r.review_date BETWEEN ? AND ?"
        )
        q += _in("r.neighbourhood", hoods, p)
        q += " ORDER BY random() LIMIT 3"
        df = con.execute(q, p).df()
    finally:
        con.close()
    return df


@st.cache_data(ttl=600)
def load_recommendation_data(start: date, end: date, hoods: tuple) -> pd.DataFrame:
    """Themes joined with review metadata — passed to rules engine."""
    con = _conn()
    try:
        p: list = [start, end]
        q = (
            "SELECT t.theme_label, r.review_date, r.neighbourhood "
            "FROM review_themes t "
            "JOIN clean_reviews r ON t.review_id = r.review_id "
            "WHERE r.review_date BETWEEN ? AND ?"
        )
        q += _in("r.neighbourhood", hoods, p)
        df = con.execute(q, p).df()
    finally:
        con.close()
    return df


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="VisitEdinburgh Insight",
        page_icon="🏰",
        layout="wide",
    )
    st.title("🏰 VisitEdinburgh Insight")

    # ── Filter options ────────────────────────────────────────────────────
    opts = load_filter_options()
    d_min: date = opts["date_min"]
    d_max: date = opts["date_max"]

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Filters")
        date_range = st.slider(
            "Date range",
            min_value=d_min, max_value=d_max,
            value=(d_min, d_max),
            format="MMM YYYY",
        )
        start, end = date_range

        sel_hoods = tuple(st.multiselect(
            "Neighbourhood", options=opts["hoods"],
            default=[], placeholder="All neighbourhoods",
        ))
        sel_cats = tuple(st.multiselect(
            "POI category", options=opts["cats"],
            default=[], placeholder="All categories",
        ))

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab_ov, tab_th, tab_re = st.tabs(["Overview", "Themes", "Recommendations"])

    # ══ Overview ══════════════════════════════════════════════════════════
    with tab_ov:
        kpis   = load_kpis(start, end, sel_hoods, sel_cats)
        poi_df = load_poi_layer(start, end, sel_hoods, sel_cats)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total reviews",        f"{kpis['total_reviews']:,}")
        c2.metric("Total POIs",           f"{kpis['total_pois']:,}")
        c3.metric("Date range",           kpis["date_range"])
        c4.metric("Avg reviews / month",  f"{kpis['avg_per_month']:,.0f}")

        st.subheader("POI Map — radius scaled by mention count")
        if poi_df.empty:
            st.info("No POIs match the current filters.")
        else:
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=poi_df,
                get_position=["longitude", "latitude"],
                get_radius="log_radius",
                radius_min_pixels=3,    # quietest café still visible
                radius_max_pixels=30,   # busiest landmark won't engulf city
                get_fill_color="[r, g, b, a]",
                pickable=True,
                auto_highlight=True,
            )
            view = pdk.ViewState(
                latitude=55.953, longitude=-3.188,
                zoom=12, pitch=0,
            )
            st.pydeck_chart(pdk.Deck(
                layers=[layer],
                initial_view_state=view,
                tooltip={"text": "{canonical_name}\n{category}\nMentions: {mention_count}"},
                map_style="light",
            ))

        st.subheader("Top 10 mentioned POIs")
        top10 = poi_df.head(10)[["canonical_name", "category", "mention_count"]].copy()
        top10.columns = ["POI", "Category", "Mentions"]
        st.dataframe(top10, use_container_width=True, hide_index=True)

    # ══ Themes ════════════════════════════════════════════════════════════
    with tab_th:
        monthly_df = load_theme_monthly(start, end, sel_hoods)
        hood_df    = load_theme_by_neighbourhood(start, end, sel_hoods)

        # Stacked area chart
        st.subheader("Theme volume over time")
        if monthly_df.empty:
            st.info("No theme data for the current filters.")
        else:
            monthly_df["month"] = pd.to_datetime(monthly_df["month"])
            fig_area = px.area(
                monthly_df, x="month", y="volume", color="theme_label",
                labels={
                    "month": "Month", "volume": "Reviews",
                    "theme_label": "Theme",
                },
            )
            fig_area.update_layout(height=380, legend_title_text="Theme")
            st.plotly_chart(fig_area, use_container_width=True)

        # Top theme per neighbourhood — horizontal bar
        st.subheader("Top theme by neighbourhood")
        if hood_df.empty:
            st.info("No neighbourhood data for the current filters.")
        else:
            top_per_hood = (
                hood_df.sort_values("volume", ascending=False)
                .groupby("neighbourhood", as_index=False)
                .first()
                .nlargest(15, "volume")
            )
            fig_bar = px.bar(
                top_per_hood,
                x="volume", y="neighbourhood", color="theme_label",
                orientation="h",
                labels={
                    "volume": "Reviews", "neighbourhood": "",
                    "theme_label": "Top Theme",
                },
            )
            fig_bar.update_layout(
                height=440,
                yaxis={"categoryorder": "total ascending"},
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Sample reviews
        st.subheader("Sample reviews by theme")
        theme_options = (
            sorted(monthly_df["theme_label"].unique().tolist())
            if not monthly_df.empty else []
        )
        if theme_options:
            chosen = st.selectbox("Select a theme", options=theme_options)
            samples = load_sample_reviews(chosen, start, end, sel_hoods)
            if samples.empty:
                st.info("No reviews found for this theme / filter.")
            else:
                for _, row in samples.iterrows():
                    st.markdown(f"> {row['snippet']}…")
                    st.caption(f"Listing {int(row['listing_id'])}")
                    st.divider()

    # ══ Recommendations ═══════════════════════════════════════════════════
    with tab_re:
        st.subheader("Insight Recommendations")
        st.caption(
            "Rule-based cards derived from theme and POI signals in the "
            "filtered dataset. Widen the date range for more rules to fire."
        )

        themes_for_rules = load_recommendation_data(start, end, sel_hoods)

        if themes_for_rules.empty:
            st.info("No data for the current filters — try widening the date range.")
        else:
            cards = get_recommendations(themes_for_rules)
            if not cards:
                st.info(
                    "No rules fired for the current selection. "
                    "Try widening the date range or resetting neighbourhood filters."
                )
            else:
                cols = st.columns(2)
                for i, card in enumerate(cards):
                    with cols[i % 2]:
                        with st.container(border=True):
                            st.markdown(f"**{card['title']}**")
                            st.write(card["suggestion"])
                            badge = (
                                f"  •  📍 {card['neighbourhood']}"
                                if card.get("neighbourhood") else ""
                            )
                            st.caption(
                                f"{card['evidence_text']}"
                                f"  •  **{card['evidence_numbers']:,}** reviews"
                                f"{badge}"
                            )


if __name__ == "__main__":
    main()
