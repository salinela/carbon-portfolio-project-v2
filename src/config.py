from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"

# static project:
START_DATE = "2018-01-01"
END_DATE = "2024-12-31"

# live project:
# START_DATE = "2018-01-01"
# END_DATE = datetime.date.today().strftime("%Y-%m-%d")


CARBON_TIERS = [
    "ets_high",      # ETS-regulated + high carbon intensity
    "ets_medium",    # ETS-regulated + medium carbon intensity  
    "ets_low",       # ETS-regulated + low carbon intensity
    "non_ets"        # Not ETS-regulated (regardless of intensity)
]

TIER_THRESHOLDS = {
    "high_cutoff": None,    # 66th percentile of ETS firms CI
    "medium_cutoff": None,  # 33rd percentile of ETS firms CI
}