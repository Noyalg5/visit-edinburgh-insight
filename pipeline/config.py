from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
DB_PATH: Path = PROJECT_ROOT / "db" / "visit.duckdb"

EDINBURGH_BBOX: tuple[float, float, float, float] = (55.92, -3.25, 55.98, -3.13)  # (south, west, north, east)

INSIDE_AIRBNB_LISTINGS_URL: str = (
    "https://data.insideairbnb.com/united-kingdom/scotland/edinburgh"
    "/2025-09-21/data/listings.csv.gz"
)
INSIDE_AIRBNB_REVIEWS_URL: str = (
    "https://data.insideairbnb.com/united-kingdom/scotland/edinburgh"
    "/2025-09-21/data/reviews.csv.gz"
)
INSIDE_AIRBNB_NEIGHBOURHOODS_URL: str = (
    "https://data.insideairbnb.com/united-kingdom/scotland/edinburgh"
    "/2025-09-21/visualisations/neighbourhoods.geojson"
)

OVERPASS_URL: str = "https://overpass.private.coffee/api/interpreter"  # overpass-api.de returns 406 from this network
