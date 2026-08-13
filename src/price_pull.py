# %% Ticker prep + price pull + load into prices table
# Wraps your batch_download_strict_min_points (UNCHANGED) with:
#   1. exchange -> yahoo-suffix mapping + dot->dash prep
#   2. storage into the prices table (idempotent via INSERT OR IGNORE)
import pandas as pd
from src.db import connect

# --- exchange -> (suffix, on_yahoo) : your map, in-universe exchanges -----
eu_exchange_to_suffix = {
    "Boerse Frankfurt": (".F", True), "Wiener Boerse": (".VI", True),
    "Boerse Berlin": (".BE", True), "Boerse Duesseldorf": (".DU", True),
    "Boerse Hamburg": (".HM", True), "Boerse Munchen": (".MU", True),
    "Boerse Stuttgart": (".SG", True), "XETRA Stock Exchange": (".DE", True),
    "London Stock Exchange": (".L", True),
    "Borsa Italiana - MTA (Mercato Telematico Azionario)": (".MI", True),
    "Euronext Paris": (".PA", True),
    "New York Stock Exchange (NYSE)": ("", True),
    "Euronext Brussels": (".BR", True), "Budapest Stock Exchange": (".BD", True),
    "Warsaw Stock Exchange": (".WA", True), "Bucharest Stock Exchange": (".RO", True),
    "Prague Stock Exchange": (".PR", True), "Euronext Amsterdam": (".AS", True),
    "Bolsa de Madrid": (".MC", True),
    "Bolsa de Comercio de Buenos Aires": (".BA", True),
    "Santiago Stock Exchange": (".SN", True), "Mercado Continuo Espanol": (".MC", True),
    "Johannesburg Stock Exchange": (".JO", True), "Nasdaq OMX - Copenhagen": (".CO", True),
    "Swiss Exchange (SWX)": (".SW", True), "NASDAQ National Market": ("", True),
    "Nasdaq OMX - Stockholm": (".ST", True), "Euronext Dublin": (".IR", True),
    "Euronext Lisbon": (".LS", True), "NASDAQ OMX PHLX": ("", True),
    "Nasdaq OMX - Helsinki": (".HE", True), "Athens Stock Exchange": (".AT", True),
    "Oslo Bors": (".OL", True), "Nasdaq OMX - Riga": (".RG", True),
    "Hong Kong Stock Exchange": (".HK", True),
    "Australian Securities Exchange": (".AX", True),
    "Istanbul Stock Exchange": (".IS", True), "Nasdaq OMX - Vilnius": (".VS", True),
    "Tel Aviv Stock Exchange": (".TA", True), "Nasdaq OMX - Tallinn": (".TL", True),
    "Nasdaq OMX - Iceland": (".IC", True), "TSX Venture Exchange": (".V", True),
    "Stock Exchange of Thailand": (".BK", True),
    "Aquis Stock Exchange": (None, False), "Pink Sheets Grey Market": (None, False),
    "OTC Pink Market": (None, False), "OTC Bulletin Board": (None, False),
    "Bulgarian Stock Exchange": (None, False), "Bern Stock Exchange": (None, False),
    "Boerse Hannover": (None, False), "Bolsa Mexicana de Valores": (None, False),
    "Bolsa de Barcelona": (None, False), "Luxembourg Stock Exchange": (None, False),
    "Bolsa de Valores de Bilbao": (None, False), "Bolsa de Valencia": (None, False),
    "Kazakhstan Stock Exchange": (None, False), "Nordic Growth Market (NGM)": (None, False),
    "Moscow Exchange MICEX - RTS": (None, False), "A2X": (None, False),
    "Ljubljana Stock Exchange": (None, False), "Zagreb Stock Exchange": (None, False),
    "PFTS Stock Exchange": (None, False), "Ukrainian Exchange": (None, False),
    "Bolsa de Valores de Lima": (None, False), "Cyprus Stock Exchange": (None, False),
    "Bratislava Stock Exchange": (None, False), "Malta Stock Exchange": (None, False),
    "Mercado Alternativo Bursatil": (None, False), "Spotlight Stock Market": (None, False),
    "Norvegian OTC": (None, False), "Euronext Milan": (".MI", True),
    "LSE": (".L", True),

    # --- CIQ short-code / variant aliases (same exchanges, different strings) ---
    "XTRA": (".DE", True), "ENXTPA": (".PA", True), "ENXTAM": (".AS", True),
    "ENXTBR": (".BR", True), "ENXTLS": (".LS", True), "BIT": (".MI", True),
    "OM": (".ST", True), "OB": (".OL", True), "WBAG": (".VI", True),
    "HLSE": (".HE", True), "SWX": (".SW", True), "WSE": (".WA", True),
    "BME": (".MC", True), "ATSE": (".AT", True), "AIM": (".L", True),
    "ASX": (".AX", True), "Toronto Stock Exchange": (".TO", True),
    "Oslo Axess Stock Exchange": (".OL", True),
    "Mercado Continuo Español": (".MC", True),   # accented variant of existing key
    "NYSE": ("", True), "NYSE MKT": ("", True), "NASDAQGS": ("", True),
    "NASDAQ Capital Market": ("", True),
    "NASDAQ/NGS (Global Select Market)": ("", True),
    "NASDAQ/NMS (Global Market)": ("", True),
    "DUSE": (".DU", True), "HMSE": (".HM", True), "CPSE": (".CO", True),
    "ICSE": (".IC", True), "BVB": (".RO", True), "BUSE": (".BD", True),

    #--- Spanish markets mapping:
    "Bolsa de Barcelona": (".MC", True),
    "Bolsa de Valores de Bilbao": (".MC", True),
    "Bolsa de Valencia": (".MC", True)
}

# suffix drives currency -> single source of truth, keys off build_yahoo_symbol's output
suffix_to_currency = {
    "": "USD", ".F":"EUR",".VI":"EUR",".BE":"EUR",".DU":"EUR",".HM":"EUR",".MU":"EUR",
    ".SG":"EUR",".DE":"EUR",".L":"GBp",".MI":"EUR",".PA":"EUR",".BR":"EUR",".BD":"HUF",
    ".WA":"PLN",".RO":"RON",".PR":"CZK",".AS":"EUR",".MC":"EUR",".BA":"ARS",".SN":"CLP",
    ".JO":"ZAc",".CO":"DKK",".SW":"CHF",".ST":"SEK",".IR":"EUR",".LS":"EUR",".HE":"EUR",
    ".AT":"EUR",".OL":"NOK",".RG":"EUR",".HK":"HKD",".AX":"AUD",".IS":"TRY",".VS":"EUR",
    ".TA":"ILA",".TL":"EUR",".IC":"ISK",".V":"CAD",".BK":"THB",".TO": "CAD"
}  # GBp/ZAc/ILA are minor units (pence/cents/agorot) -> /100 handled at FX read-time

def exchange_currency(exchange):
    suffix, on_yahoo = eu_exchange_to_suffix.get(exchange, (None, False))
    return suffix_to_currency.get(suffix) if on_yahoo else None



def build_yahoo_symbol(ticker, exchange):
    """Bare Orbis ticker + exchange -> Yahoo symbol, or None if unmappable."""
    if pd.isna(ticker):
        return None
    suffix, on_yahoo = eu_exchange_to_suffix.get(exchange, (None, False))
    if not on_yahoo:
        return None
    
    # block to prep ticker symbols:
    bare = str(ticker).strip()
    bare = bare.rstrip('.')            # trailing-dot artifact:  UU.  -> UU
    bare = bare.replace(' ', '-')      # space class-separator:  SEB A -> SEB-A
    bare = bare.replace('.', '-')      # dot class-separator:    HOLM.B -> HOLM-B
    return f"{bare}{suffix}"


def prep_symbols(con):
    """Master rows -> yahoo_symbol + currency, with coverage report."""
    m = pd.read_sql("SELECT company_id, ticker, exchange, universe "
                    "FROM master_company_list WHERE ticker IS NOT NULL", con)
    m['yahoo_symbol'] = m.apply(
        lambda r: build_yahoo_symbol(r['ticker'], r['exchange']), axis=1)
    m['currency'] = m['exchange'].map(exchange_currency)
    ok = m[m['yahoo_symbol'].notna()].copy()
    unmapped = m[m['yahoo_symbol'].isna()]
    print(f"tickers total     : {len(m):,}")
    print(f"mapped to yahoo   : {len(ok):,}")
    print(f"unmapped exchange : {len(unmapped):,}")
    print("currency mix      :")
    print(ok['currency'].value_counts().to_string())   # eyeball GBp / exotic units
    if len(unmapped):
        print("Unmapped Exchange:\n",unmapped['exchange'].value_counts().head(10).to_string())
    return ok, unmapped

def load_prices(con, price_df, symbol_to_id, symbol_to_currency):
    """Write raw OHLCV + currency into `prices`, idempotent on (company_id, date)."""
    p = price_df.rename(columns={
        'Date':'date','Open':'open','High':'high','Low':'low',
        'Close':'close','Volume':'volume'}).copy()
    p['company_id'] = p['Ticker'].map(symbol_to_id)
    p['currency']   = p['Ticker'].map(symbol_to_currency)
    p = p.dropna(subset=['company_id'])
    p['date'] = pd.to_datetime(p['date']).dt.strftime('%Y-%m-%d')
    p['source'] = 'yfinance'
    cols = ['company_id','date','open','high','low','close','volume','currency','source']
    p = p[[c for c in cols if c in p.columns]].dropna(subset=['date'])   # 'Adj Close' dropped here
    rows = list(p.itertuples(index=False, name=None))
    ph = ','.join('?' * len(p.columns))
    con.executemany(
        f"INSERT OR IGNORE INTO prices ({','.join(p.columns)}) VALUES ({ph})", rows)
    con.commit()
    print(f"prices upserted: {len(p):,} rows, {p['company_id'].nunique():,} companies")

def build_symbol_coverage(con, ok, unmapped):
    """Persist the full mapping->load funnel to symbol_coverage (one row per company).
    Loaded-status is read from `prices` (ground truth), not the pull's report."""
    loaded = set(pd.read_sql("SELECT DISTINCT company_id FROM prices", con)['company_id'])
    keep = ['company_id','ticker','exchange','yahoo_symbol','currency','universe']

    a = ok[keep].copy()
    a['status'] = a['company_id'].apply(
        lambda c: 'mapped_loaded' if c in loaded else 'mapped_no_data')

    b = unmapped[keep].copy()
    b['status'] = 'unmapped_exchange'

    nt = pd.read_sql("SELECT company_id, ticker, exchange, universe "
                     "FROM master_company_list WHERE ticker IS NULL", con)
    nt['yahoo_symbol'] = None; nt['currency'] = None; nt['status'] = 'no_ticker'

    cols = ['company_id','ticker','exchange','yahoo_symbol','currency','status','universe']
    cov = pd.concat([a, b, nt], ignore_index=True)[cols]

    con.executemany(
        f"INSERT OR REPLACE INTO symbol_coverage ({','.join(cols)}) "
        f"VALUES ({','.join('?'*len(cols))})",
        list(cov.itertuples(index=False, name=None)))
    con.commit()
    print(f"symbol_coverage: {len(cov):,} rows")
    print(cov['status'].value_counts().to_string())
    return cov

# %%
