# Daily EUR FX for read-time conversion of raw local prices to EUR.
# rate_per_eur = units of `currency` per 1 EUR (yfinance 'EUR{CCY}=X').
# Convert: eur = local / rate_per_eur   (minor units e.g. GBp: divide price by 100 first).
import pandas as pd
import yfinance as yf
import time

MINOR_UNIT = {"GBp": "GBP", "ZAc": "ZAR", "ILA": "ILS"}  # -> major used for the FX pair


def currencies_in_universe(con):
    """Distinct price currencies, normalised to the major unit; EUR excluded."""
    cur = pd.read_sql("SELECT DISTINCT currency FROM prices "
                      "WHERE currency IS NOT NULL", con)['currency'].tolist()
    majors = {MINOR_UNIT.get(c, c) for c in cur}
    return sorted(majors - {"EUR"})


def fetch_fx(currencies, start="2013-01-01", end="2026-06-30", pause=2):
    """EUR->CCY daily rates. Returns long df: date, currency, rate_per_eur."""
    frames, missing = [], []
    for ccy in currencies:
        try:
            df = yf.download(f"EUR{ccy}=X", start=start, end=end,
                             auto_adjust=False, progress=False)
            if df.empty:
                missing.append(ccy); time.sleep(pause); continue
            s = df['Close']
            if hasattr(s, 'columns'):           # single-ticker MultiIndex guard
                s = s.iloc[:, 0]
            out = s.reset_index()
            out.columns = ['date', 'rate_per_eur']
            out['currency'] = ccy
            frames.append(out)
        except Exception as e:
            print(f"  [ERROR] EUR{ccy}=X: {e}"); missing.append(ccy)
        time.sleep(pause)
    if missing:
        print(f"no FX for: {missing}")
    if not frames:
        return pd.DataFrame(columns=['date','currency','rate_per_eur'])
    fx = pd.concat(frames, ignore_index=True)
    fx['date'] = pd.to_datetime(fx['date']).dt.strftime('%Y-%m-%d')
    return fx[['date','currency','rate_per_eur']].dropna()


def load_fx(con, fx_df):
    """INSERT OR IGNORE into fx_rates, idempotent on (date, currency)."""
    if fx_df.empty:
        print("no FX rows"); return
    rows = list(fx_df.itertuples(index=False, name=None))
    con.executemany("INSERT OR IGNORE INTO fx_rates (date, currency, rate_per_eur) "
                    "VALUES (?, ?, ?)", rows)
    con.commit()
    print(f"fx upserted: {len(fx_df):,} rows, {fx_df['currency'].nunique()} currencies")