# Per-ticker splits + dividends via yfinance .actions, for adjust-on-read.
# Splits: ratio (2:1 -> 2.0). Dividends: per-share, LOCAL currency (same as prices).
# Pulled only for eligible firms; keyed on company_id via the symbol->id map.
import pandas as pd
import yfinance as yf
import time
import os

def fetch_actions(symbol_to_id, batch_pause=0.5, progress_every=25,
                  checkpoint_csv="actions_checkpoint.csv"):
    """Pull splits+dividends per ticker, with checkpointing + heartbeat.
    Resumes from checkpoint_csv on re-run; skips already-fetched firms."""
    done = set()
    if os.path.exists(checkpoint_csv):
        prev = pd.read_csv(checkpoint_csv, dtype=str)
        done = set(prev['company_id'].unique())
        print(f"resuming: {len(done):,} firms already done, skipping them")

    todo = {s: c for s, c in symbol_to_id.items() if c not in done}
    total = len(todo)
    print(f"to fetch: {total:,} firms\n")

    rows, missing, n_hits = [], [], 0
    for i, (sym, cid) in enumerate(todo.items(), start=1):
        try:
            act = yf.Ticker(sym).history(period="max", actions=True, timeout=10)
            if act is not None and not act.empty:
                act = act.reset_index()
                for _, r in act.iterrows():
                    d = pd.to_datetime(r['Date']).strftime('%Y-%m-%d')
                    div = r.get('Dividends', 0) or 0
                    spl = r.get('Stock Splits', 0) or 0
                    if spl and spl > 0:
                        rows.append((cid, d, 'split', float(spl))); n_hits += 1
                    if div and div > 0:
                        rows.append((cid, d, 'dividend', float(div))); n_hits += 1
        except Exception:
            missing.append(sym)

        # heartbeat: overwrite one line every firm so you see live progress
        print(f"\r  {i}/{total}  ({100*i//total}%)  actions so far: {n_hits:,}",
              end="", flush=True)

        # checkpoint every progress_every firms
        if i % progress_every == 0 and rows:
            pd.DataFrame(rows, columns=['company_id','date','action_type','value']) \
              .to_csv(checkpoint_csv, mode='a',
                      header=not os.path.exists(checkpoint_csv), index=False)
            rows = []

        time.sleep(batch_pause)

    if rows:   # final flush
        pd.DataFrame(rows, columns=['company_id','date','action_type','value']) \
          .to_csv(checkpoint_csv, mode='a',
                  header=not os.path.exists(checkpoint_csv), index=False)
    print()  # newline after the \r heartbeat
    if missing:
        print(f"failed for {len(missing)} symbols (first 10): {missing[:10]}")

    df = pd.read_csv(checkpoint_csv, dtype={'company_id':str,'date':str,'action_type':str}) \
         if os.path.exists(checkpoint_csv) else \
         pd.DataFrame(columns=['company_id','date','action_type','value'])
    if not df.empty:
        df['value'] = df['value'].astype(float)
    print(f"\ntotal: {len(df):,} actions: "
          f"{(df['action_type']=='split').sum():,} splits, "
          f"{(df['action_type']=='dividend').sum():,} dividends")
    return df


def load_actions(con, actions_df):
    """INSERT OR IGNORE into corporate_actions, idempotent on (company_id, date, action_type)."""
    if actions_df.empty:
        print("no actions"); return
    rows = list(actions_df.itertuples(index=False, name=None))
    con.executemany(
        "INSERT OR IGNORE INTO corporate_actions (company_id, date, action_type, value) "
        "VALUES (?, ?, ?, ?)", rows)
    con.commit()
    print(f"actions upserted: {len(actions_df):,} rows, "
          f"{actions_df['company_id'].nunique():,} companies")