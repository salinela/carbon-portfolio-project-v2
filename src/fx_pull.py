# Daily EUR FX rates, for read-time conversion of raw local prices to EUR.
#
# STORED: rate_per_eur = units of `currency` per 1 EUR  (from yfinance 'EUR{CCY}=X').
# READ-TIME (feature stage, later): eur_price = local_price / rate_per_eur
#   - minor-unit currencies (GBp/ZAc/ILA) also need local_price / 100 first
#   - forward-fill rate onto stock dates that FX weekends/holidays miss
# We store the MAJOR-unit rate (GBP/ZAR/ILS); the /100 is applied at read-time.

import pandas as pd
import yfinance as yf
import time

MINOR_UNIT = {"GBp": "GBP", "ZAc": "ZAR", "ILA": "ILS"}  # price-tag -> FX-pair currency


def currencies_in_universe(con):
    """Distinct non-EUR price currencies, normalised to the major unit."""
    cur = pd.read_sql(
        "SELECT DISTINCT currency FROM prices WHERE currency IS NOT NULL", con
    )['currency'].tolist()
    majors = {MINOR_UNIT.get(c, c) for c in cur}   # GBp->GBP, else unchanged
    return sorted(majors - {"EUR"})                # EUR is the base, no rate needed


def fetch_fx(currencies, start="2013-01-01", end="2026-06-30", pause=2):
    """Download EUR->CCY daily rates. Returns long df: date, currency, rate_per_eur."""
    frames, missing = [], []
    for ccy in currencies:
        try:
            df = yf.download(f"EUR{ccy}=X", start=start, end=end,
                             auto_adjust=False, progress=False)
            if df is None or df.empty:
                missing.append(ccy); time.sleep(pause); continue
            close = df['Close']
            if isinstance(close, pd.DataFrame):        # single-ticker shape guard
                close = close.iloc[:, 0]
            out = close.reset_index()
            out.columns = ['date', 'rate_per_eur']
            out['currency'] = ccy
            frames.append(out)
        except Exception as e:
            print(f"  [ERROR] EUR{ccy}=X: {e}"); missing.append(ccy)
        time.sleep(pause)
    if missing:
        print(f"no FX for: {missing}")
    if not frames:
        return pd.DataFrame(columns=['date', 'currency', 'rate_per_eur'])
    fx = pd.concat(frames, ignore_index=True)
    fx['date'] = pd.to_datetime(fx['date']).dt.strftime('%Y-%m-%d')
    return fx[['date', 'currency', 'rate_per_eur']].dropna()


def load_fx(con, fx_df):
    """INSERT OR IGNORE into fx_rates, idempotent on (date, currency)."""
    if fx_df.empty:
        print("no FX rows"); return
    rows = list(fx_df.itertuples(index=False, name=None))
    con.executemany(
        "INSERT OR IGNORE INTO fx_rates (date, currency, rate_per_eur) VALUES (?, ?, ?)",
        rows)
    con.commit()
    print(f"fx upserted: {len(fx_df):,} rows, {fx_df['currency'].nunique()} currencies")