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
def load_raw_prices(con, start=None, end=None, companies=None) -> pd.DataFrame:
    """Raw OHLCV in local currency (long). start/end are ISO 'YYYY-MM-DD'.
    companies: optional list of company_id to restrict to (used by batched build)."""
    q = "SELECT company_id, date, open, high, low, close, volume, currency FROM prices"
    clauses, params = [], []
    if start:
        clauses.append("date >= ?"); params.append(start)
    if end:
        clauses.append("date <= ?"); params.append(end)
    if companies is not None:
        clauses.append(f"company_id IN ({','.join('?' * len(companies))})")
        params.extend(companies)
    if clauses:
        q += " WHERE " + " AND ".join(clauses)
    df = pd.read_sql(q, con, params=params or None, parse_dates=["date"])
    return df.sort_values(["company_id", "date"]).reset_index(drop=True)


def _split_factors(con, companies, dates_index) -> pd.DataFrame | None:
    """
    Wide frame (date x company) of the cumulative FUTURE split factor F[t] =
    product of split ratios strictly after t. adj_price[t] = raw[t] / F[t]
    makes the series continuous backward through splits. None if no splits.
    """
    sp = pd.read_sql(
        "SELECT company_id, date, value AS ratio FROM corporate_actions "
        "WHERE action_type = 'split'",
        con, parse_dates=["date"],
    )
    if sp.empty:
        return None
    ratio = (sp.pivot_table(index="date", columns="company_id", values="ratio", aggfunc="prod")
               .reindex(index=dates_index, columns=companies))
    ratio = ratio.fillna(1.0)
    # F[t] = product of ratios for split dates > t  == reverse-cumprod, shifted up one row
    incl = ratio.iloc[::-1].cumprod().iloc[::-1]      # product over [t .. last] inclusive
    future = incl.shift(-1).fillna(1.0)               # strictly after t
    return future


def load_prices(con, start=None, end=None, adjust=True, companies=None) -> dict[str, pd.DataFrame]:
    """
    Return split-adjusted WIDE OHLCV frames plus currency, ready for vectorised
    rolling. Keys: 'open','high','low','close','volume','currency'(Series).
    Volume is adjusted inversely to price so dollar-volume stays continuous.
    companies: optional list to restrict to (batched build).
    """
    raw = load_raw_prices(con, start, end, companies)
    dates = pd.DatetimeIndex(sorted(raw["date"].unique()))
    comps = sorted(raw["company_id"].unique())

    wide = {c: raw.pivot(index="date", columns="company_id", values=c).reindex(dates, columns=comps)
            for c in ("open", "high", "low", "close", "volume")}
    ccy = raw.groupby("company_id")["currency"].last().reindex(comps)

    if adjust:
        F = _split_factors(con, comps, dates)
        if F is not None:
            for c in ("open", "high", "low", "close"):
                wide[c] = wide[c] / F
            wide["volume"] = wide["volume"] * F      # shares scale up on a split
    wide["currency"] = ccy
    return wide


def load_fx(con) -> pd.DataFrame | None:
    fx = pd.read_sql("SELECT date, currency, rate_per_eur FROM fx_rates", con, parse_dates=["date"])
    if fx.empty:
        return None
    return fx.pivot(index="date", columns="currency", values="rate_per_eur")


def _load_factor_level(con, factor, price_type="close") -> pd.Series | None:
    """Raw level series for one market_factors factor. None if absent."""
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
# Fraction of a rolling window that must be populated before a value is emitted.
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
        rng = (hh - ll).replace(0, np.nan)          # flat window -> NaN, not /0 -> inf
        k = 100 * (close - ll) / rng
        out[f"stoch_k_{n}"] = k
        out[f"stoch_d_{n}"] = k.rolling(3, min_periods=3).mean()
    return out


def f_williams(close, high, low):
    out = {}
    for n in W["williams"]:
        hh = high.rolling(n, min_periods=_mp(n)).max()
        ll = low.rolling(n, min_periods=_mp(n)).min()
        rng = (hh - ll).replace(0, np.nan)          # flat window -> NaN
        out[f"williams_r_{n}"] = -100 * (hh - close) / rng
    return out


def f_bollinger(close):
    out = {}
    for n in W["boll"]:
        mid = close.rolling(n, min_periods=_mp(n)).mean()
        sd = close.rolling(n, min_periods=_mp(n)).std()
        upper, lower = mid + 2 * sd, mid - 2 * sd
        band = (upper - lower).replace(0, np.nan)   # zero-width band (sd=0) -> NaN
        out[f"boll_pctb_{n}"] = (close - lower) / band              # %B, position in band
        out[f"boll_bw_{n}"] = band / mid.replace(0, np.nan)         # bandwidth, vol proxy
    return out


def f_macd(close):
    close_safe = close.replace(0, np.nan)           # guards exact-zero price only; see note
    out = {}
    for (fast, slow, sig) in W["macd"]:
        ema_f = close.ewm(span=fast, min_periods=_mp(slow)).mean()
        ema_s = close.ewm(span=slow, min_periods=_mp(slow)).mean()
        macd = ema_f - ema_s
        signal = macd.ewm(span=sig, min_periods=_mp(slow)).mean()
        tag = f"{fast}_{slow}_{sig}"
        out[f"macd_{tag}"] = macd / close_safe                     # normalised by price
        out[f"macd_hist_{tag}"] = (macd - signal) / close_safe
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
                         long_format=True, resample=None, companies=None, me_dates=None):
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

    companies : optional list of company_id to restrict to (used by the batched
                driver). me_dates: optional explicit month-end DatetimeIndex so
                every batch shares ONE grid (else each batch would derive its own).

    Deployment: call with a short `start` for incremental serving; the trailing
    windows need history, so pass start >= (serve_date - ~300 trading days).
    For the full universe on limited RAM, prefer build_price_features_batched().
    """
    px = load_prices(con, start, end, adjust=True, companies=companies)
    close, high, low = px["close"], px["high"], px["low"]
    volume, ccy = px["volume"], px["currency"]
    logclose = np.log(close)
    ret = log_returns(close)
    fx_wide = load_fx(con)

    feats: dict[str, pd.DataFrame] = {}
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
        me = me_dates if me_dates is not None else month_end_dates(close.index)
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
    out["value"] = out["value"].replace([np.inf, -np.inf], np.nan)   # backstop: no inf survives
    return out[["company_id", "date", "signal_name", "value"]].dropna(subset=["value"])


def build_price_features_batched(con, start=None, end=None, batch_size=600,
                                 resample="M", verbose=True, **kwargs):
    """
    Memory-safe full-universe build: process companies in batches and concatenate
    the (month-end) long outputs. Peak RAM ~= one batch, not the whole universe.

    Only meaningful with resample="M" (daily long across all firms is enormous).
    One shared month-end grid is derived once from the full price calendar so
    every batch lands on identical dates. batch_size 600 keeps the SQL IN-clause
    under SQLite's default variable limit; lower it if RAM is very tight.
    """
    if resample != "M":
        raise ValueError("batched build is intended for resample='M'")

    comps = pd.read_sql("SELECT DISTINCT company_id FROM prices", con)["company_id"].tolist()
    dcond = []
    if start: dcond.append(f"date >= '{start}'")
    if end:   dcond.append(f"date <= '{end}'")
    dq = "SELECT DISTINCT date FROM prices" + (" WHERE " + " AND ".join(dcond) if dcond else "")
    all_dates = pd.to_datetime(pd.read_sql(dq, con)["date"])
    me = month_end_dates(all_dates)                       # ONE grid for every batch

    batches = [comps[i:i + batch_size] for i in range(0, len(comps), batch_size)]
    parts = []
    for i, batch in enumerate(batches, 1):
        part = build_price_features(con, start=start, end=end, resample="M",
                                    companies=batch, me_dates=me, **kwargs)
        parts.append(part)
        if verbose:
            print(f"  batch {i}/{len(batches)}  ({len(batch)} firms)  ->  {len(part):,} rows")
    return pd.concat(parts, ignore_index=True)


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


def forward_return_label(con, horizon=1, kind="log", start=None, end=None):
    """
    Minimal month-end forward-return TARGET for EDA (feature-vs-target work).
    At each month-end t, the label is the return realised over the NEXT `horizon`
    months -- i.e. what the trailing features at t should predict.

    kind='log' (default) matches the modelling convention; 'arith' available too.
    This is price return (ex-dividend), aligned to the same month-end grid as the
    features. The PRODUCTION label (total return incl. dividends, plus the CV
    embargo) is built later in modeling.py -- this is only for exploratory work.

    Returns long: company_id, date, fwd_ret.  Merge onto the feature panel on
    (company_id, date). Rows in the last `horizon` months are NaN (no future) and
    dropped.
    """
    close = load_prices(con, start, end, adjust=True)["close"]
    m = close.reindex(month_end_dates(close.index))          # month-end wide close
    if kind == "log":
        fwd = np.log(m).shift(-horizon) - np.log(m)          # shift(-h): pull FUTURE back to t
    elif kind == "arith":
        fwd = m.shift(-horizon) / m - 1.0
    else:
        raise ValueError("kind must be 'log' or 'arith'")
    s = fwd.stack()
    s.index = s.index.set_names(["date", "company_id"])
    out = s.rename("fwd_ret").reset_index()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out[["company_id", "date", "fwd_ret"]].dropna(subset=["fwd_ret"])


# ===========================================================================
# 6. OPTIONAL POST-PROCESSING FILTERS  (off by default -- prune on demand)
# ===========================================================================
def eligible_ids(con, statuses=("mapped_loaded",), extra_where=None):
    """
    company_ids passing symbol_coverage eligibility. Default keeps status
    'mapped_loaded'. If your symbol_coverage has density / n_real columns, pass
    e.g. extra_where="AND density >= 0.90 AND n_real >= 250".
    Use with filter_panel(panel, keep_ids=eligible_ids(con)).
    """
    q = "SELECT company_id FROM symbol_coverage WHERE status IN ({})".format(
        ",".join("?" * len(statuses)))
    if extra_where:
        q += " " + extra_where
    return pd.read_sql(q, con, params=list(statuses))["company_id"].tolist()


def filter_panel(panel, keep_ids=None, drop_ids=None, max_missing_frac=None,
                 drop_inf_firms=True, verbose=True):
    """
    OPTIONAL post-processing on the WIDE panel (index=[company_id, date], one
    column per signal). Nothing is removed unless you ask -- keeps the full
    universe by default so you can prune only when a run needs it.

    keep_ids         : keep only these company_ids (e.g. eligible_ids(con))
    drop_ids         : drop these company_ids
    max_missing_frac : drop a firm whose mean missingness across signals exceeds this
    drop_inf_firms   : drop any firm that still has an inf cell (backstop; the
                       denominator guards should already prevent this)
    """
    cid = panel.index.get_level_values("company_id")
    keep = pd.Series(True, index=panel.index)
    log = {}
    if keep_ids is not None:
        m = cid.isin(set(keep_ids)); log["not_in_keep_ids"] = int((~m).sum()); keep &= m
    if drop_ids is not None:
        m = ~cid.isin(set(drop_ids)); log["in_drop_ids"] = int((~m).sum()); keep &= m
    if drop_inf_firms:
        inf_firms = pd.Index(cid[np.isinf(panel.to_numpy()).any(axis=1)]).unique()
        m = ~cid.isin(set(inf_firms)); log["inf_firms"] = len(inf_firms); keep &= m
    if max_missing_frac is not None:
        firm_miss = panel.isna().mean(axis=1).groupby(cid).mean()
        bad = firm_miss[firm_miss > max_missing_frac].index
        m = ~cid.isin(set(bad)); log["too_missing"] = len(bad); keep &= m
    out = panel[keep]
    if verbose:
        n0, n1 = cid.nunique(), out.index.get_level_values("company_id").nunique()
        print(f"filter_panel {log} | firms {n0}->{n1} | rows {len(panel):,}->{len(out):,}")
    return out


# Example (notebook orchestration):
#   import sqlite3, pandas as pd
#   con = sqlite3.connect("carbon.db")
#   month_end = build_price_features(con, start="2013-01-01", resample="M")   # modeling grid
#   label     = forward_return_label(con)                                     # EDA target
#   # optional persist:  month_end.to_parquet("data/processed/features_month_end.parquet")
