"""SQLite helpers for carbon-portfolio-project-v2.

Reusable DB utilities. Import into notebooks / main code:

    from src.db import init_db, load_table, connect

Default paths are relative to the PROJECT ROOT. Pass explicit paths (or
override the module constants) if you run from elsewhere.
"""
import sqlite3
from pathlib import Path
import pandas as pd

DB_PATH     = 'data/carbon.db'
SCHEMA_PATH = 'sql/schema.sql'


def connect(db_path=DB_PATH):
    """Open a connection with foreign keys ON (SQLite defaults them OFF)."""
    con = sqlite3.connect(db_path)
    con.execute('PRAGMA foreign_keys = ON;')
    return con


def init_db(db_path=DB_PATH, schema_path=SCHEMA_PATH, drop_existing=False):
    """Create the database from schema.sql and return an open connection."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    if drop_existing and Path(db_path).exists():
        Path(db_path).unlink()
    con = connect(db_path)
    con.executescript(Path(schema_path).read_text())
    con.commit()
    return con


def load_table(con, df, table, colmap=None, replace=True):
    """
    Write `df` to `table`, keeping only columns the table declares.

    colmap  : rename df columns -> table columns before filtering
    replace : DELETE existing rows first (True) or append (False)
    Prints row count and any declared columns with no data.
    """
    out = df.rename(columns=colmap) if colmap else df.copy()

    table_cols = [r[1] for r in con.execute(f'PRAGMA table_info({table})')]
    keep = [c for c in table_cols if c in out.columns]
    missing = [c for c in table_cols if c not in out.columns]
    out = out[keep]

    for c in out.columns:                       # SQLite has no date/bool type
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime('%Y-%m-%d')
        elif pd.api.types.is_bool_dtype(out[c]):
            out[c] = out[c].astype(int)

    if replace:
        con.execute(f'DELETE FROM {table}')
    out.to_sql(table, con, if_exists='append', index=False)
    con.commit()

    print(f"  {table:22s} {len(out):>7,} rows"
          + (f"   (no data for: {', '.join(missing)})" if missing else ""))
