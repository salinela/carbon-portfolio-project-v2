# price pull functions from y_finance

import pandas as pd
import yfinance as yf
import time


# working version:
def batch_download_strict_min_points(
    tickers,
    start="2013-01-01",
    end=None,
    auto_adjust=False,
    batch_size=50,
    pause=20,
    output_csv=None,
    no_data_csv=None,
    min_count=6
):
    tickers = list(tickers)
    total = len(tickers)
    all_batches = []
    seen_tickers = set()  # Track all tickers returned

    print(f"Total tickers to process: {total}")
    for i in range(0, total, batch_size):
        batch = tickers[i:i+batch_size]
        print(f"Downloading batch {i} to {i+len(batch)-1} / {total-1}")
        try:
            df = yf.download(
                batch,
                start=start,
                end=end,
                auto_adjust=auto_adjust,
                progress=False
            )
            if df.empty:
                print(f"  [NO DATA] Batch {batch}")
                time.sleep(pause)
                continue

            # If only one ticker, DataFrame is not MultiIndex; handle that
            if len(batch) == 1 or (df.columns.nlevels == 1):
                df['Ticker'] = batch[0]
                df = df.reset_index()
                all_batches.append(df)
                seen_tickers.add(batch[0])
                
            else:
                # MultiIndex columns: field, ticker
                df_stacked = df.stack(level=1, future_stack=True).reset_index()
                df_stacked = df_stacked.rename(columns={'level_1': 'Ticker'})
                all_batches.append(df_stacked)
                seen_tickers.update(df_stacked['Ticker'].unique())
        except Exception as e:
            print(f"  [ERROR] Batch {batch}: {e}")
        time.sleep(pause)

    # All tickers ever requested
    all_requested = set(tickers)
    # All tickers that returned any data
    found_tickers = seen_tickers
    # Tickers never returned (no rows at all)
    never_seen = sorted(all_requested - found_tickers)

    if all_batches:
        result_df = pd.concat(all_batches, ignore_index=True)
        # Count non-NaN 'Close' values per ticker
        close_counts = result_df.groupby('Ticker')['Close'].count()
        # Tickers with < min_count non-NaN closes
        too_short = close_counts[close_counts < min_count].index.tolist()
        print(f"\nTickers with < {min_count} non-NaN Close values: {too_short}")
        # Final filtered DataFrame
        filtered_df = result_df[~result_df['Ticker'].isin(too_short + never_seen)].copy()
    else:
        filtered_df = pd.DataFrame()
        too_short = []

    if output_csv and not filtered_df.empty:
        filtered_df.to_csv(output_csv, index=False)
        print(f"Saved filtered data to {output_csv}")
    if no_data_csv:
        pd.DataFrame({
            'never_seen': pd.Series(never_seen),
            'too_short': pd.Series(too_short)
        }).to_csv(no_data_csv, index=False)
        print(f"Saved missing/too short tickers to {no_data_csv}")

    print(f"\nFinal: {len(filtered_df['Ticker'].unique())} tickers with >= {min_count} data, "
          f"{len(too_short)} with too little data, {len(never_seen)} never returned any data.")
    return filtered_df, too_short, never_seen
