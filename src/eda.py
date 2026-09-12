"""EDA assembly + reusable figure/frame functions — carbon-portfolio-project-v2.
Every function takes data in and returns a DataFrame or Plotly figure, no print
side effects, so the same functions serve the notebook and the Streamlit tabs.
"""

import numpy as np
import pandas as pd


# Phase 0: Loading data
def build_meta(con):
    """One row per company: static attributes. Join key for panel + composition.
    Returns (meta indexed by company_id, diag dict)."""
    mcl = pd.read_sql_query(
        """
        SELECT company_id, company_name, universe, sector, nace_code,
               country, listing_status, n_ets_entities, has_emissions,
               bvd_id_number
        FROM master_company_list
        """, con)
    n_base = len(mcl)

    orb = pd.read_sql_query(
        """
        SELECT bvd_id_number, bvd_sectors,
               nace_code AS orbis_nace_code, nace_description AS orbis_nace_desc
        FROM orbis_core
        """, con)

    meta = mcl.merge(orb, on="bvd_id_number", how="left")
    assert len(meta) == n_base, f"orbis join inflated rows: {len(meta)} != {n_base}"

    sector_nulls_master = int(meta["sector"].isna().sum())
    meta["sector"] = meta["sector"].fillna(meta["bvd_sectors"])   # backfill
    sector_nulls_final = int(meta["sector"].isna().sum())

    cov = pd.read_sql_query(
        "SELECT company_id, status AS coverage_status FROM symbol_coverage", con)
    meta = meta.merge(cov, on="company_id", how="left")
    assert len(meta) == n_base, f"coverage join inflated rows: {len(meta)} != {n_base}"
    meta["eligible"] = (meta["coverage_status"] == "mapped_loaded").astype(int)

    meta = meta.set_index("company_id")
    diag = {
        "n_companies": n_base,
        "by_universe": meta["universe"].value_counts().to_dict(),
        "eligible_n": int(meta["eligible"].sum()),
        "sector_nulls_master": sector_nulls_master,
        "sector_nulls_after_backfill": sector_nulls_final,
        "country_nulls": int(meta["country"].isna().sum()),
        "bvd_nulls": int(meta["bvd_id_number"].isna().sum()),
        "coverage_status_counts":
            meta["coverage_status"].value_counts(dropna=False).to_dict(),
    }
    return meta, diag


def build_firm_year(con):
    """One row per company-year: time-varying numerics. Feeds tiering + the grid.
    Raw fiscal-year values only — the as-of shift (FY-Y usable 1 Jul Y+1) is
    applied at panel-merge time, never here. Returns (firm_year, diag dict)."""
    emis = pd.read_sql_query(
        "SELECT company_id, year, scope1_emissions, source FROM company_emissions", con)
    n_emis = len(emis)

    fund = pd.read_sql_query(
        """ SELECT company_id, year,            
                   revenue, ebitda, net_income,
                   total_assets,
                   total_debt, total_common_equity
            FROM fundamentals
        """, con)

    fy = emis.merge(fund, on=["company_id", "year"], how="left")
    assert len(fy) == n_emis, f"fundamentals join inflated rows: {len(fy)} != {n_emis}"

    # raw intensity = tCO2 per EUR '000. NB05 tiered on intensity_w (weighted/
    # winsorised?) — reconcile before recomputing tiers in Phase 1.
    with np.errstate(divide="ignore", invalid="ignore"):
        fy["intensity"] = fy["scope1_emissions"] / fy["revenue"]
    fy.loc[~np.isfinite(fy["intensity"]), "intensity"] = np.nan

    diag = {
        "n_rows": len(fy),
        "n_companies": fy["company_id"].nunique(),
        "year_range": (int(fy["year"].min()), int(fy["year"].max())),
        "by_source": fy["source"].value_counts().to_dict(),
        "revenue_nulls": int(fy["revenue"].isna().sum()),
        "intensity_nulls": int(fy["intensity"].isna().sum()),
    }
    return fy, diag


# Phase 1:

## section A: tiering company emissions intensity status (low, medium and high):

def _tercile(g, min_n=6):
    """Rank-based terciles within a group; NaN if fewer than min_n non-nulls.
    Rank-first breaks ties so qcut always gets three equal-count bins."""
    if g.notna().sum() < min_n:
        return pd.Series(np.nan, index=g.index)
    return pd.qcut(g.rank(method="first"), 3, labels=["low", "medium", "high"])


def compute_tiers(firm_year, nace1, intensity_col="intensity", min_n=6):
    """Six-way, time-varying carbon tier. Terciles of intensity WITHIN
    source x year x nace1; fallback WITHIN source x year; residual _untiered.
    Derived, never stored. `nace1`: Series/dict company_id -> 1-digit NACE.
    Returns (firm_year + nace1 + carbon_tier, diag)."""
    fy = firm_year.copy()
    fy["nace1"] = fy["company_id"].map(nace1)
    fy["carbon_tier"] = pd.Series(pd.NA, index=fy.index, dtype="object")

    for src, prefix in {"ets_registry": "ets", "trucost": "non_ets"}.items():
        m = (fy["source"] == src) & fy[intensity_col].notna()

        # primary: within year x nace1
        t = fy.loc[m].groupby(["year", "nace1"])[intensity_col].transform(
            lambda g: _tercile(g, min_n))
        fy.loc[m, "carbon_tier"] = (prefix + "_" + t.astype("string")).where(
            t.notna(), f"{prefix}_untiered")

        # fallback: still-untiered rows, within year only
        mu = m & (fy["carbon_tier"] == f"{prefix}_untiered")
        t2 = fy.loc[mu].groupby("year")[intensity_col].transform(
            lambda g: _tercile(g, min_n))
        fy.loc[mu, "carbon_tier"] = (prefix + "_" + t2.astype("string")).where(
            t2.notna(), f"{prefix}_untiered")

    untiered = ["ets_untiered", "non_ets_untiered"]
    diag = {
        "tier_counts": fy["carbon_tier"].value_counts(dropna=False).to_dict(),
        "nace1_nulls": int(fy["nace1"].isna().sum()),
        "n_untiered": int(fy["carbon_tier"].isin(untiered).sum()),
        "n_tiered": int(fy["carbon_tier"].notna().sum()
                        - fy["carbon_tier"].isin(untiered).sum()),
    }
    return fy, diag


# Section B — carbon tier-based portfolio return helper

"""
tiers are annual (fiscal year), the panel is monthly;
seach company-month inherits the tier of the latest fiscal year usable at that date under your 1-July-Y+1 rule 


Read-time only  Second decision: the label is log fwd_ret, but a portfolio return is a cross-sectional average, and averaging must happen on arithmetic returns (log isn't additive across firms) — so the helper does expm1 before averaging, honoring your log-for-signals / arithmetic-for-portfolios convention. Equal-weighted for now; a weight= param can add cap-weighting later without touching callers.
"""

def attach_tier_asof(panel, firm_year, tier_col="carbon_tier"):
    """
    Tells each company-month which basket it was in at that date, using the 1-July lag.

    Add as-of carbon_tier to a (company_id, date) panel. FY-Y usable from
    1-Jul-(Y+1); read-time only, never stored. Carbon-blind firms stay NaN."""

    # panel is merged features + monthly log returns:
    p = panel.reset_index()
    dt = pd.to_datetime(p["date"])

    # assigning financial year:
    p["_fy"] = np.where(dt.dt.month >= 7, dt.dt.year - 1, dt.dt.year - 2) # np.where(condition, output if true, output if false)
    key = firm_year[["company_id", "year", tier_col]].rename(columns={"year": "_fy"})
    n0 = len(p)
    out = p.merge(key, on=["company_id", "_fy"], how="left")
    assert len(out) == n0, f"tier merge inflated rows: {len(out)} != {n0}"
    return out.drop(columns="_fy").set_index(["company_id", "date"])



def tier_portfolio_returns(panel_with_tier, ret_col="fwd_ret",
                           tier_col="carbon_tier", log_input=True, min_names=5):
    """Equal-weighted tier return series (date x tier), arithmetic. Converts log
    forward returns to arithmetic before cross-sectional averaging. Tier-dates
    with < min_names firms are NaN'd (thin cross-section)."""
    df = (panel_with_tier.reset_index()[["date", tier_col, ret_col]]
          .dropna(subset=[tier_col, ret_col]))
    df["_r"] = np.expm1(df[ret_col]) if log_input else df[ret_col]
    g = df.groupby(["date", tier_col])["_r"]
    mean = g.mean().unstack(tier_col)
    n = g.size().unstack(tier_col)
    mean = mean.where(n >= min_names)
    mean.index = pd.to_datetime(mean.index)
    return mean.sort_index()


# Detection scan (flags any month-end where label coverage is abnormally low relative to features):

def label_coverage_scan(features_long, label_long, flag_frac=0.5):
    """Per month-end: firms with features vs firms with a non-NaN label. Flags
    months whose label/feature ratio is < flag_frac of the median ratio.
    Note: the final `horizon` month(s) legitimately have no label (no future)."""
    f = features_long.groupby("date")["company_id"].nunique().rename("n_features")
    l = label_long.groupby("date")["company_id"].nunique().rename("n_label")
    cov = pd.concat([f, l], axis=1).fillna(0)
    cov["ratio"] = cov["n_label"] / cov["n_features"].where(cov["n_features"] > 0)
    cov["flag"] = cov["ratio"] < flag_frac * cov["ratio"].median()
    return cov.sort_values("ratio")