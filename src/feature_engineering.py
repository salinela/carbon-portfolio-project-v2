"""
feature_engineering.py  —  price-derived (technical) features for carbon-portfolio-project-v2

Design contract
---------------
* Read-time. Nothing here is stored unless the notebook explicitly persists the
  long-format output to `signals`. Every function is a pure transform of what's
  in `carbon.db`.
* Vectorised. Rolling windows run on WIDE frames (date index x company columns)
  so one call computes an indicator for all ~8,288 companies at once.
* Deployment-ready. The same functions serve backtest (bulk history) and the
  live serving layer (pass a connection with a shorter date range).
* Leakage-safe by construction. Every window is TRAILING (uses data up to and
  including the observation date only). Month-end sampling and the as-of merge
  of carbon/fundamentals happen LATER, in the matrix assembler — not here.

Graceful degradation (only prices/fundamentals/emissions are loaded so far)
--------------------------------------------------------------------------
* corporate_actions empty  -> split adjustment is a no-op (returns raw prices).
* fx_rates empty           -> EUR conversion falls back to LOCAL currency; the
                              money features (dollar_volume, amihud) are then NOT
                              cross-currency comparable. A warning is emitted.
* market_factors empty     -> market features (beta, corr, idio_vol, regime) are
                              skipped automatically.

Scale-invariance note
---------------------
Returns and ratio indicators need no FX. Only dollar_volume and amihud are money-
denominated; those are converted to EUR (with GBp/ZAc/ILA /100 subunit handling).
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

TRADING_DAYS = 252

# ---------------------------------------------------------------------------
# WINDOWS  — edit here; every function reads from this block.
# ---------------------------------------------------------------------------
W = {
    # trend
    "sma_fast": 50, "sma_slow": 200,
    # momentum
    "mom_lookback": 252, "mom_skip": 21,   # 12-1 momentum: last ~12m, skip last month
    "reversal": 21,                          # 1-month short-term reversal
    # realized volatility (multiple horizons)
    "vol": (21, 63, 252),
    # extremes
    "max_ret": 21,                           # MAX = largest 1-day return last month
    "high_52w": 252,                         # proximity to 52-week high
    "drawdown": 252,                         # drawdown vs trailing peak
    # liquidity (money-denominated -> EUR)
    "dollar_vol": 21,
    "amihud": 21,
    # oscillators: (short, long) pairs — both kept; both may carry info
    "rsi": (14, 50),
    "stoch": (14, 63),                       # %K windows; %D = 3-period SMA of %K
    "williams": (14, 63),
    "boll": (20, 60),                        # SMA/std window for Bollinger
    "macd": ((12, 26, 9), (24, 52, 18)),     # (fast, slow, signal) short & long
    # market condition
    "beta": (63, 252),
    "market_corr": 63,
    "idio_vol": 63,
    "market_regime": 21,                     # market trailing return / vol window
}

# Subunit currencies quoted in 1/100 of the major unit -> (major, divisor)
SUBUNIT = {"GBp": ("GBP", 100.0), "ZAc": ("ZAR", 100.0), "ILA": ("ILS", 100.0)}


# ===========================================================================
# 1. LOADERS + PRICE ADJUSTMENT
# ===========================================================================
def load_raw_prices(con, start=None, end=None) -> pd.DataFrame:
    """Raw OHLCV in local currency (long).
    start/end are ISO 'YYYY-MM-DD'.
    Called in load_prices() as helper
    """

    q = "SELECT company_id, date, open, high, low, close, volume, currency FROM prices"
    clauses = []
    if start:
        clauses.append(f"date >= '{start}'")
    if end:
        clauses.append(f"date <= '{end}'")
    if clauses:
        q += " WHERE " + " AND ".join(clauses)
    df = pd.read_sql(q, con, parse_dates=["date"])
    return df.sort_values(["company_id", "date"]).reset_index(drop=True)


def _split_factors(con, companies, dates_index) -> pd.DataFrame | None:
    """
    objective: we rescale old prices so the series is continuous through the splits
    e.g., [100,100,50,50] to [50,50,50,50] via F = [2,2,1,1]

    Wide frame (date x company) of the cumulative FUTURE split factor F[t] =
    product of split ratios strictly after t. 
    
    adj_price[t] = raw[t] / F[t]

    Called in load_prices() as helper
    """

    # read split actions from corporate_actions table:
    sp = pd.read_sql(
        "SELECT company_id, date, value AS ratio FROM corporate_actions "
        "WHERE action_type = 'split'",
        con, parse_dates=["date"],
    )
    if sp.empty:
        return None
    
    # wide table
    ratio = (sp.pivot_table(index="date", columns="company_id", values="ratio", aggfunc="prod")
               .reindex(index=dates_index, columns=companies)) # add all dates within time range as rows
    
    # Derive F
    ratio = ratio.fillna(1.0) # unsplit days have split of 1.0 

    # F[t] = product of ratios for split dates > t  == reverse-cumprod, shifted up one row and then fillna with 1
    incl = ratio.iloc[::-1].cumprod().iloc[::-1]      # product over [t .. last] inclusive
    future = incl.shift(-1).fillna(1.0)               # strictly after t

    return future


def load_prices(con, start=None, end=None, adjust=True) -> dict[str, pd.DataFrame]:
    """
    Return split-adjusted WIDE OHLCV frames plus currency, ready for vectorised
    rolling. Keys: 'open','high','low','close','volume','currency'(Series).
    Volume is adjusted inversely to price so dollar-volume stays continuous.
    Called in main orchestator
    """
    raw = load_raw_prices(con, start, end)
    dates = pd.DatetimeIndex(sorted(raw["date"].unique()))
    comps = sorted(raw["company_id"].unique())

    wide = {c: raw.pivot(index="date", columns="company_id", values=c).reindex(dates, columns=comps)
            for c in ("open", "high", "low", "close", "volume")}
    ccy = raw.groupby("company_id")["currency"].last().reindex(comps)

    if adjust:
        F = _split_factors(con, comps, dates)
        if F is not None:
            for c in ("open", "high", "low", "close"):

                # adjust raw prices based on derived split factors table:
                wide[c] = wide[c] / F
            wide["volume"] = wide["volume"] * F      # shares scale up on a split
    wide["currency"] = ccy
    return wide


def load_fx(con) -> pd.DataFrame | None:
    """
    Called in main orchestrator
    """
    fx = pd.read_sql("SELECT date, currency, rate_per_eur FROM fx_rates", con, parse_dates=["date"])
    if fx.empty:
        return None
    return fx.pivot(index="date", columns="currency", values="rate_per_eur")

# Market Factor subsection:
def _load_factor_level(con, factor, price_type="close") -> pd.Series | None:
    """Raw level series for one market_factors factor. None if absent.
    Called in load_market_return() and load_factor_return() as helper
    """

    # need to specify market factor type (brent, eua, natgas, stoxx600, us10y, vix, wti) and price type (OHLVC+ adjusted Close)
    mk = pd.read_sql(
        "SELECT date, price FROM market_factors WHERE factor = ? AND price_type = ? ORDER BY date",
        con, params=(factor, price_type), parse_dates=["date"],
    )
    if mk.empty:
        return None
    return mk.set_index("date")["price"].astype(float)


def _factor_return(level, kind, index=None):
    """
    Transform a level series to a daily factor return, aligned to `index`.
    kind='log'  -> log return  (price series: eua, brent, stoxx600)
    kind='diff' -> first difference in native units (yield series: us10y)
    Level is calendar-aligned and ffilled BEFORE transforming so a holiday gap
    doesn't null a whole rolling window.

    Called in load_market_return() as helper
    """
    if level is None:
        return None
    if index is not None:
        level = level.reindex(level.index.union(index)).sort_index().ffill().reindex(index)
    return (np.log(level).diff() if kind == "log" else level.diff())


def load_market_return(con, factor="stoxx600", price_type="close", index=None) -> pd.Series | None:
    """Daily market (price-index) log return; None if the factor is absent."""


    lvl = _load_factor_level(con, factor, price_type)


    return None if lvl is None else _factor_return(lvl, "log", index).rename("market_ret")


# Factors that get firm-level rolling betas (chosen: EUA carbon, Brent oil, US 10y rates).
# 'kind' picks the return transform: log for prices, diff for the yield.
FACTOR_SPECS = {
    "eua":   {"price_type": "close", "kind": "log"},   # EU carbon allowance price
    "brent": {"price_type": "close", "kind": "log"},   # European oil benchmark
    "us10y": {"price_type": "close", "kind": "diff"},  # yield -> Δyield, not a return
}


def load_factor_return(con, factor, price_type="close", kind="log", index=None) -> pd.Series | None:
    lvl = _load_factor_level(con, factor, price_type)
    return None if lvl is None else _factor_return(lvl, kind, index)


def _eur_close(close_wide, ccy, fx_wide):
    """Convert adjusted local close to EUR (subunit /100 then / rate_per_eur)."""
    if fx_wide is None:
        warnings.warn("fx_rates empty -> money features left in LOCAL currency, "
                      "NOT cross-currency comparable.", stacklevel=2)
        return close_wide
    fx = fx_wide.reindex(close_wide.index).ffill()
    out = close_wide.copy()
    for comp in close_wide.columns:
        c = ccy.get(comp)
        if c is None or c == "EUR":
            continue
        div = 1.0
        if c in SUBUNIT:
            c, div = SUBUNIT[c]
        rate = fx[c] if c in fx.columns else np.nan
        out[comp] = (close_wide[comp] / div) / rate
    return out


# ===========================================================================
# 2. PRIMITIVES
# ===========================================================================
def log_returns(close_wide) -> pd.DataFrame:
    """Daily log returns. Reusable base for vol, momentum, beta, MAX, amihud."""
    return np.log(close_wide).diff()


# ===========================================================================
# 3. INDICATORS  (each returns one or more WIDE frames, date x company)
# ===========================================================================
# Fraction of a rolling window that must be populated before a value is emitted (instead of full 252 trading days)
# The wide pivot is built on the UNION of all firms' trading calendars, so every
# column carries holiday-gap NaNs from other exchanges. Requiring the *full*
# window (min_periods=_mp(w)) then makes long windows (200/252d) unsatisfiable. 0.70
# tolerates the gaps while staying trailing-only (no leakage).


MIN_PERIODS_FRAC = 0.70

def _mp(w):
    """min_periods for a window of length w: a fraction of it, floor of 2."""
    return max(2, int(round(MIN_PERIODS_FRAC * w)))


def f_trend(close):
    sma_f = close.rolling(W["sma_fast"], min_periods=_mp(W["sma_fast"])).mean()
    sma_s = close.rolling(W["sma_slow"], min_periods=_mp(W["sma_slow"])).mean()
    return {
        "trend_dist_sma200": close / sma_s - 1.0,      # level vs long trend
        "trend_sma50_200":  sma_f / sma_s - 1.0,       # golden/death cross, continuous
    }


def f_momentum(logclose):
    lb, sk = W["mom_lookback"], W["mom_skip"]
    return {
        "mom_12_1":   logclose.shift(sk) - logclose.shift(lb),   # 12-1 momentum
        "reversal_1m": logclose - logclose.shift(W["reversal"]),  # sign expected NEGATIVE
    }


def f_realized_vol(ret):
    out = {}
    for w in W["vol"]:
        out[f"rvol_{w}"] = ret.rolling(w, min_periods=_mp(w)).std() * np.sqrt(TRADING_DAYS)
    return out


def f_extremes(close, ret):
    w = W["max_ret"]
    hi = close.rolling(W["high_52w"], min_periods=_mp(W["high_52w"])).max()
    peak = close.rolling(W["drawdown"], min_periods=_mp(W["drawdown"])).max()
    return {
        "max_ret_1m":   ret.rolling(w, min_periods=_mp(w)).max(),  # lottery proxy; sign NEGATIVE
        "high_52w_prox": close / hi,                                # 0..1, near 1 = near high
        "drawdown":     close / peak - 1.0,                         # <=0
    }


def f_liquidity(close, volume, ccy, fx_wide, ret):
    eur_close = _eur_close(close, ccy, fx_wide)
    dvol = eur_close * volume                                       # EUR traded per day
    w_dv, w_am = W["dollar_vol"], W["amihud"]
    amihud = (ret.abs() / dvol.replace(0, np.nan))
    return {
        "dollar_vol": np.log1p(dvol.rolling(w_dv, min_periods=_mp(w_dv)).mean()),
        "amihud":     amihud.rolling(w_am, min_periods=_mp(w_am)).mean() * 1e6,
    }


# ---- oscillators (short + long) -------------------------------------------
def f_rsi(close):
    delta = close.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    out = {}
    for n in W["rsi"]:
        ag = gain.ewm(alpha=1 / n, min_periods=_mp(n)).mean()           # Wilder smoothing
        al = loss.ewm(alpha=1 / n, min_periods=_mp(n)).mean()
        out[f"rsi_{n}"] = 100 - 100 / (1 + ag / al)
    return out


def f_stochastic(close, high, low):
    out = {}
    for n in W["stoch"]:
        ll = low.rolling(n, min_periods=_mp(n)).min()
        hh = high.rolling(n, min_periods=_mp(n)).max()
        k = 100 * (close - ll) / (hh - ll)
        out[f"stoch_k_{n}"] = k
        out[f"stoch_d_{n}"] = k.rolling(3, min_periods=3).mean()
    return out


def f_williams(close, high, low):
    out = {}
    for n in W["williams"]:
        hh = high.rolling(n, min_periods=_mp(n)).max()
        ll = low.rolling(n, min_periods=_mp(n)).min()
        out[f"williams_r_{n}"] = -100 * (hh - close) / (hh - ll)
    return out


def f_bollinger(close):
    out = {}
    for n in W["boll"]:
        mid = close.rolling(n, min_periods=_mp(n)).mean()
        sd = close.rolling(n, min_periods=_mp(n)).std()
        upper, lower = mid + 2 * sd, mid - 2 * sd
        out[f"boll_pctb_{n}"] = (close - lower) / (upper - lower)   # %B, position in band
        out[f"boll_bw_{n}"] = (upper - lower) / mid                 # bandwidth, vol proxy
    return out


def f_macd(close):
    out = {}
    for (fast, slow, sig) in W["macd"]:
        ema_f = close.ewm(span=fast, min_periods=_mp(slow)).mean()
        ema_s = close.ewm(span=slow, min_periods=_mp(slow)).mean()
        macd = ema_f - ema_s
        signal = macd.ewm(span=sig, min_periods=_mp(slow)).mean()
        tag = f"{fast}_{slow}_{sig}"
        out[f"macd_{tag}"] = macd / close                          # normalised by price
        out[f"macd_hist_{tag}"] = (macd - signal) / close
    return out


# ---- market condition ------------------------------------------------------
def _rolling_beta(ret, factor_ret, w):
    """Rolling beta of every firm (cols of ret) to one factor return series."""
    fr = factor_ret.reindex(ret.index)
    mean_i = ret.rolling(w, min_periods=_mp(w)).mean()
    mean_f = fr.rolling(w, min_periods=_mp(w)).mean()
    cov = ret.mul(fr, axis=0).rolling(w, min_periods=_mp(w)).mean().sub(mean_i.mul(mean_f, axis=0))
    var_f = fr.pow(2).rolling(w, min_periods=_mp(w)).mean() - mean_f.pow(2)
    return cov.div(var_f, axis=0)


def _rolling_corr(ret, factor_ret, w):
    fr = factor_ret.reindex(ret.index)
    mean_i = ret.rolling(w, min_periods=_mp(w)).mean()
    mean_f = fr.rolling(w, min_periods=_mp(w)).mean()
    cov = ret.mul(fr, axis=0).rolling(w, min_periods=_mp(w)).mean().sub(mean_i.mul(mean_f, axis=0))
    std_i = ret.rolling(w, min_periods=_mp(w)).std()
    std_f = fr.rolling(w, min_periods=_mp(w)).std()
    return cov.div(std_i.mul(std_f, axis=0))


def f_market(ret, market_ret):
    """
    beta / correlation / idiosyncratic vol to the equity market (cross-sectional),
    plus market-state regime columns (constant across firms on a date -> useful
    only in a POOLED model, wiped out by within-date normalisation).
    """
    if market_ret is None:
        return {}
    out = {}
    for w in W["beta"]:
        out[f"beta_{w}"] = _rolling_beta(ret, market_ret, w)

    corr = _rolling_corr(ret, market_ret, W["market_corr"])
    out[f"market_corr_{W['market_corr']}"] = corr

    wv = W["idio_vol"]
    total = ret.rolling(wv, min_periods=_mp(wv)).std() * np.sqrt(TRADING_DAYS)
    corr_v = corr if wv == W["market_corr"] else _rolling_corr(ret, market_ret, wv)
    out[f"idio_vol_{wv}"] = total * np.sqrt((1 - corr_v.pow(2)).clip(lower=0))

    wr = W["market_regime"]
    rm = market_ret.reindex(ret.index)
    peak = (1 + rm.fillna(0)).cumprod()
    regime = {
        "mkt_ret": rm.rolling(wr, min_periods=_mp(wr)).sum(),
        "mkt_vol": rm.rolling(wr, min_periods=_mp(wr)).std() * np.sqrt(TRADING_DAYS),
        "mkt_drawdown": peak / peak.cummax() - 1.0,
    }
    frame = ret * 0.0                       # template: broadcast state to every firm
    for name, s in regime.items():
        out[name] = frame.add(s, axis=0)
    return out


def f_factor_betas(ret, con, windows=None):
    """
    Firm-level rolling betas to each factor in FACTOR_SPECS (EUA, Brent, us10y).
    These DO vary cross-sectionally, so they are legitimate features:
      eua_beta_*   = market-priced carbon-risk sensitivity (the thesis channel)
      brent_beta_* = oil/energy exposure
      us10y_beta_* = rate-duration exposure (beta to Δyield)
    Absent factors are skipped silently.
    """
    windows = windows or W["beta"]
    out = {}
    for factor, spec in FACTOR_SPECS.items():
        fr = load_factor_return(con, factor, spec["price_type"], spec["kind"], index=ret.index)
        if fr is None:
            continue
        for w in windows:
            out[f"{factor}_beta_{w}"] = _rolling_beta(ret, fr, w)
    return out


# ===========================================================================
# 4. ORCHESTRATOR
# ===========================================================================
def build_price_features(con, start=None, end=None, factor="stoxx600",
                         include_market=True, include_factor_betas=True,
                         long_format=True, resample=None):
    """
    Compute every price feature and return either a dict of wide frames or a
    tidy long DataFrame (company_id, date, signal_name, value) matching `signals`.

    resample=None : keep DAILY rows (huge for the full universe -> use only on a
                    subset / short window, e.g. the dashboard's single-ticker view).
    resample="M"  : reduce each wide frame to the panel's month-end trading dates
                    BEFORE melting. This is the modeling grid and the memory-safe
                    path: the expensive melt only ever sees ~monthly rows. Uniform
                    dates across firms (a firm not trading on the panel month-end
                    date is NaN there and dropped), which is what the matrix wants.

    Deployment: call with a short `start` for incremental serving; the trailing
    windows need history, so pass start >= (serve_date - ~300 trading days).
    """
    px = load_prices(con, start, end, adjust=True)
    close, high, low = px["close"], px["high"], px["low"]
    volume, ccy = px["volume"], px["currency"]
    logclose = np.log(close)
    ret = log_returns(close)
    fx_wide = load_fx(con)

    # initialize empty dictionary
    feats: dict[str, pd.DataFrame] = {}

    # update dictionary with specific technical indicator functions and relevant input params:
    feats.update(f_trend(close))
    feats.update(f_momentum(logclose))
    feats.update(f_realized_vol(ret))
    feats.update(f_extremes(close, ret))
    feats.update(f_liquidity(close, volume, ccy, fx_wide, ret))
    feats.update(f_rsi(close))
    feats.update(f_stochastic(close, high, low))
    feats.update(f_williams(close, high, low))
    feats.update(f_bollinger(close))
    feats.update(f_macd(close))

    if include_market:
        feats.update(f_market(ret, load_market_return(con, factor=factor, index=ret.index)))
    if include_factor_betas:
        feats.update(f_factor_betas(ret, con))

    if resample == "M":
        me = month_end_dates(close.index)
        feats = {name: wide.reindex(me) for name, wide in feats.items()}
    elif resample is not None:
        raise ValueError("resample must be None or 'M'")

    if not long_format:
        return feats

    parts = []
    for name, wide in feats.items():
        s = wide.stack()
        s.index = s.index.set_names(["date", "company_id"])
        parts.append(s.rename("value").reset_index().assign(signal_name=name))
    out = pd.concat(parts, ignore_index=True)
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    
    return out[["company_id", "date", "signal_name", "value"]].dropna(subset=["value"])


# ===========================================================================
# 5. MONTH-END SAMPLERS  (feed the matrix assembler in the modeling step)
# ===========================================================================
def month_end_dates(index) -> pd.DatetimeIndex:
    """The last actual trading date in each calendar month across the panel."""
    idx = pd.DatetimeIndex(index)
    last = pd.Series(idx, index=idx).groupby(idx.to_period("M")).max()
    return pd.DatetimeIndex(last.values)


def to_month_end(features_long) -> pd.DataFrame:
    """
    Long-format month-end sampler: last obs per (company, signal, month).
    Use this when you already have DAILY long output; for the full build prefer
    build_price_features(..., resample="M"), which is far cheaper.
    """
    df = features_long.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["ym"] = df["date"].dt.to_period("M")
    idx = df.groupby(["company_id", "signal_name", "ym"])["date"].idxmax()
    out = df.loc[idx].drop(columns="ym")
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out.reset_index(drop=True)


# Example (notebook orchestration):
#   import sqlite3, pandas as pd
#   con = sqlite3.connect("carbon.db")
#   month_end = build_price_features(con, start="2013-01-01", resample="M")   # modeling grid
#   # optional persist:  month_end.to_sql("signals", con, if_exists="append", index=False)
