-- ============================================================================
-- carbon-portfolio-project-v2  —  SQLite schema
--
-- Grain summary (the thing to keep straight):
--   installations      one row per ETS installation      (~19,624)
--   ets_entity         one row per ETS legal entity      (6,803)
--   master_company_list one row per LISTED SECURITY      (8,288)
--
-- The chain is  installation -> ets_entity -> company.  Many installations per
-- entity, many entities per company (ENGIE holds 62).  Returns attach at the
-- company level; emissions aggregate UP to it.
--
-- SQLite has no DATE type: all dates are TEXT in ISO 'YYYY-MM-DD' form, which
-- sorts and compares correctly as text.
-- Run  PRAGMA foreign_keys = ON;  on every connection — SQLite defaults to OFF.
-- ============================================================================

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;


-- ---------------------------------------------------------------------------
-- REFERENCE TABLES
-- ---------------------------------------------------------------------------

-- The analysis universe: one row per tradeable security.
CREATE TABLE IF NOT EXISTS master_company_list (
    company_id      TEXT PRIMARY KEY,          -- ISIN, or ticker where no ISIN
    company_name    TEXT NOT NULL,
    ticker          TEXT,                      -- NULL for the 109 ISIN-only
    isin            TEXT,
    exchange        TEXT,
    country         TEXT,                      -- ISO-2
    sector          TEXT,                      -- to backfill from orbis_core
    bvd_id_number   TEXT,                      -- own bvd (EU) or parent's bvd (ETS);
                                               -- FK to orbis_core, nullable (CIQ-only
                                               -- parents have no orbis_core row)
    universe        TEXT NOT NULL CHECK (universe IN ('ETS', 'EU')),
    sp_entity_id    TEXT,                      -- CIQ permanent id
    listing_status  TEXT,
    n_ets_entities  INTEGER DEFAULT 0,
    has_emissions   INTEGER DEFAULT 0,        -- 1 if present in company_emissions
    resolution      TEXT,                      -- how the ETS link was made
    data_start_date TEXT,
    data_end_date   TEXT
);
CREATE INDEX IF NOT EXISTS ix_mcl_universe ON master_company_list(universe);
CREATE INDEX IF NOT EXISTS ix_mcl_ticker   ON master_company_list(ticker);
CREATE INDEX IF NOT EXISTS ix_mcl_country  ON master_company_list(country);
CREATE INDEX IF NOT EXISTS ix_mcl_bvd      ON master_company_list(bvd_id_number);


-- Every ETS legal entity, resolved or not. Keeps the unresolvable ones so
-- sample attrition stays auditable.
CREATE TABLE IF NOT EXISTS ets_entity (
    bvd_id_number         TEXT PRIMARY KEY,
    company_name          TEXT,
    country               TEXT,
    resolution_method     TEXT NOT NULL,
    in_portfolio_universe INTEGER NOT NULL CHECK (in_portfolio_universe IN (0,1)),
    parent_bvd_id         TEXT,
    parent_name           TEXT,
    parent_ticker         TEXT,
    parent_isin           TEXT,
    parent_country        TEXT,
    parent_status         TEXT,
    parent_delisting_date TEXT,
    matched_via           TEXT
);
CREATE INDEX IF NOT EXISTS ix_entity_method ON ets_entity(resolution_method);


-- Link table: replaces the pipe-delimited ets_entity_ids string.
-- Many entities -> one company.
CREATE TABLE IF NOT EXISTS ets_entity_company (
    bvd_id_number TEXT NOT NULL,
    company_id    TEXT NOT NULL,
    PRIMARY KEY (bvd_id_number, company_id),
    FOREIGN KEY (bvd_id_number) REFERENCES ets_entity(bvd_id_number),
    FOREIGN KEY (company_id)    REFERENCES master_company_list(company_id)
);
CREATE INDEX IF NOT EXISTS ix_eec_company ON ets_entity_company(company_id);


-- Decomposed Orbis exports, kept raw so nothing upstream is lost.
CREATE TABLE IF NOT EXISTS orbis_core (
    bvd_id_number   TEXT PRIMARY KEY,
    company_name    TEXT,
    country_iso_code TEXT,
    country         TEXT,
    nace_code       TEXT,
    nace_description TEXT,
    bvd_sectors     TEXT,
    legal_form      TEXT,
    listing_status  TEXT,
    ticker_symbol   TEXT,
    isin_number     TEXT,
    main_exchange   TEXT,
    ipo_date        TEXT,
    delisting_date  TEXT,
    incorporation_date TEXT,
    currency        TEXT,
    operating_revenue REAL,
    n_employees     INTEGER,
    last_avail_year INTEGER,
    source_table    TEXT CHECK (source_table IN ('ets', 'eu'))
);
CREATE INDEX IF NOT EXISTS ix_core_isin   ON orbis_core(isin_number);
CREATE INDEX IF NOT EXISTS ix_core_sector ON orbis_core(bvd_sectors);


-- Long format: one row per (entity, identifier type).
CREATE TABLE IF NOT EXISTS orbis_identifiers (
    bvd_id_number   TEXT NOT NULL,
    identifier_type TEXT NOT NULL,   -- lei / vat / trade_register / tin ...
    identifier_value TEXT,
    PRIMARY KEY (bvd_id_number, identifier_type),
    FOREIGN KEY (bvd_id_number) REFERENCES orbis_core(bvd_id_number)
);
CREATE INDEX IF NOT EXISTS ix_ident_value ON orbis_identifiers(identifier_value);


-- Long format: one row per (entity, owner tier).
CREATE TABLE IF NOT EXISTS orbis_ownership (
    bvd_id_number       TEXT NOT NULL,
    owner_type          TEXT NOT NULL CHECK (owner_type IN ('guo','duo','ish')),
    owner_name          TEXT,
    owner_bvd_id_number TEXT,
    owner_country       TEXT,
    owner_ticker        TEXT,
    PRIMARY KEY (bvd_id_number, owner_type),
    FOREIGN KEY (bvd_id_number) REFERENCES orbis_core(bvd_id_number)
);
CREATE INDEX IF NOT EXISTS ix_own_owner ON orbis_ownership(owner_bvd_id_number);


-- From notebook 01. Column names PROVISIONAL — adjust to the real output.
CREATE TABLE IF NOT EXISTS installations (
    installation_id  TEXT PRIMARY KEY,
    installation_name TEXT,
    bvd_id_number    TEXT,             -- NULL for the 4,244 unmatched
    account_id       TEXT,
    operator_type    TEXT CHECK (operator_type IN ('stationary','aircraft','maritime')),
    activity_type    TEXT,
    country          TEXT,
    match_method     TEXT,             -- bvd_id / registration_number / none
    opened_date      TEXT,
    closed_date      TEXT
);
CREATE INDEX IF NOT EXISTS ix_inst_bvd      ON installations(bvd_id_number);
CREATE INDEX IF NOT EXISTS ix_inst_operator ON installations(operator_type);


-- ---------------------------------------------------------------------------
-- PANEL TABLES
-- ---------------------------------------------------------------------------

-- Verified emissions / allocation per installation-year (EU ETS registry).
CREATE TABLE IF NOT EXISTS emissions (
    installation_id     TEXT NOT NULL,
    year                INTEGER NOT NULL,
    verified_emissions  REAL,
    allocated_free      REAL,
    surrendered         REAL,
    compliance_status   TEXT,
    PRIMARY KEY (installation_id, year),
    FOREIGN KEY (installation_id) REFERENCES installations(installation_id)
);
CREATE INDEX IF NOT EXISTS ix_emis_year ON emissions(year);


-- Company-year Scope 1 emissions, the COMMON metric across both universes.
-- ETS firms: verified emissions from the EU registry (aggregated to company).
-- Non-ETS: Absolute GHG Scope 1 disclosed via Trucost.
-- Intensity (emissions/revenue) is derived downstream as a signal, not stored.
CREATE TABLE IF NOT EXISTS company_emissions (
    company_id       TEXT NOT NULL,
    year             INTEGER NOT NULL,
    scope1_emissions REAL,                  -- absolute GHG Scope 1
    source           TEXT CHECK (source IN ('ets_registry', 'trucost')),
    PRIMARY KEY (company_id, year),
    FOREIGN KEY (company_id) REFERENCES master_company_list(company_id)
);
CREATE INDEX IF NOT EXISTS ix_cemis_year ON company_emissions(year);


-- Daily OHLCV. ~8,288 companies x ~1,760 trading days = ~14.6M rows.
-- Raw local-currency prices only (no adj_close): raw values never change,
-- so INSERT OR IGNORE is truly idempotent. Splits/dividends handled on read.
-- NOTE the index on date alone: walk-forward validation slices the panel
-- BY DATE across all companies, which the composite PK cannot serve.
CREATE TABLE IF NOT EXISTS prices (
    company_id  TEXT NOT NULL,
    date        TEXT NOT NULL,
    open        REAL,
    high        REAL,
    low         REAL,
    close       REAL,
    volume      REAL,
    currency    TEXT,                  -- 'EUR','GBp','SEK'...  for read-time FX
    source      TEXT,                  -- yfinance / ciq
    PRIMARY KEY (company_id, date),
    FOREIGN KEY (company_id) REFERENCES master_company_list(company_id)
);
CREATE INDEX IF NOT EXISTS ix_price_date ON prices(date);


CREATE TABLE IF NOT EXISTS fx_rates (
    date         TEXT NOT NULL,
    currency     TEXT NOT NULL,        -- major unit: 'USD','SEK','GBP'...
    rate_per_eur REAL,                 -- units of `currency` per 1 EUR
    source       TEXT DEFAULT 'yfinance',
    PRIMARY KEY (date, currency)
);


-- Fundamentals per company-period. Long format keeps it flexible as the
-- metric list grows without schema migrations.
CREATE TABLE IF NOT EXISTS fundamentals (
    company_id   TEXT NOT NULL,
    period_end   TEXT NOT NULL,
    metric       TEXT NOT NULL,
    value        REAL,
    period_type  TEXT CHECK (period_type IN ('annual','quarterly')),
    PRIMARY KEY (company_id, period_end, metric),
    FOREIGN KEY (company_id) REFERENCES master_company_list(company_id)
);
CREATE INDEX IF NOT EXISTS ix_fund_metric ON fundamentals(metric, period_end);


-- Engineered features / model inputs, long format for the same reason.
CREATE TABLE IF NOT EXISTS signals (
    company_id  TEXT NOT NULL,
    date        TEXT NOT NULL,
    signal_name TEXT NOT NULL,
    value       REAL,
    PRIMARY KEY (company_id, date, signal_name),
    FOREIGN KEY (company_id) REFERENCES master_company_list(company_id)
);
CREATE INDEX IF NOT EXISTS ix_sig_name_date ON signals(signal_name, date);


-- Symbol resolution + load funnel. Process/log data, not master data.
-- One row per company: 8,288 total = mapped_loaded + mapped_no_data + unmapped_exchange + no_ticker.
CREATE TABLE IF NOT EXISTS symbol_coverage (
    company_id   TEXT PRIMARY KEY,
    ticker       TEXT,
    exchange     TEXT,
    yahoo_symbol TEXT,
    currency     TEXT,
    status       TEXT,          -- mapped_loaded / mapped_no_data / unmapped_exchange / no_ticker
    universe     TEXT,
    FOREIGN KEY (company_id) REFERENCES master_company_list(company_id)
);

-- ---------------------------------------------------------------------------
-- CONVENIENCE VIEWS
-- ---------------------------------------------------------------------------

-- Company-year carbon exposure: the aggregation the whole chain exists for.
CREATE VIEW IF NOT EXISTS v_company_emissions AS
SELECT
    eec.company_id,
    e.year,
    SUM(e.verified_emissions) AS total_verified_emissions,
    SUM(e.allocated_free)     AS total_allocated_free,
    SUM(e.allocated_free) - SUM(e.verified_emissions) AS net_allowance_position,
    COUNT(DISTINCT i.installation_id) AS n_installations
FROM emissions e
JOIN installations i       ON i.installation_id = e.installation_id
JOIN ets_entity_company eec ON eec.bvd_id_number = i.bvd_id_number
GROUP BY eec.company_id, e.year;


-- Sample composition check.
CREATE VIEW IF NOT EXISTS v_universe_summary AS
SELECT universe, country, COUNT(*) AS n_companies,
       SUM(CASE WHEN ticker IS NULL THEN 1 ELSE 0 END) AS n_no_ticker
FROM master_company_list
GROUP BY universe, country;


