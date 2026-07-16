# cursor.md

Project guidance for Cursor when working in this repository.

## Coding Behavior

- Minimum code that solves the problem; no speculative abstractions.
- Touch only what the task requires; match existing style in `degiro_analytics/`.
- Define verifiable success criteria before multi-step work.

## Project Purpose

DeGiro portfolio analytics: connect via `degiro_connector`, persist EOD prices in SQLite, compute performance/dividends/optimization/regimes, deliver charts via script + Jupyter notebook.

## Python Environment

Use conda **`markines311`** env for this project:

```powershell
conda activate markines311
python scripts/run_degiro_analysis.py
```

Or without conda on PATH:

```powershell
C:\Users\mardom10000232\AppData\Local\anaconda3\envs\markines311\python.exe scripts/run_degiro_analysis.py
```

## Credentials & API Keys

| Item | Path |
|------|------|
| DeGiro login | `config/config.json` (copy from `mark_ines/medium/config/config.json`) |
| Market data keys | `.env` (copy from `.env.example` or `EOM/.env`: `POLYGON_API_KEY`, `EODHD_API_KEY`) |

Load env with `python-dotenv` in `degiro_analytics/settings.py`.

## EOD providers

Fetch order for closes: **EODHD → Polygon → Yahoo chart API → yfinance**.

```powershell
# Connectivity probe (free sources + optional paid keys)
python scripts/check_eod_sources.py --allow-missing-keys --write-db

# Smoke-test EODHD with their public demo token (AAPL/US only)
python scripts/check_eod_sources.py --use-eodhd-demo --allow-missing-keys
```

## Data Flow

```
degiro_connector API
        │
        ▼
data/cache/degiro_connector_data/   (account, positions, transactions CSV)
        │
        ▼
EOD providers (EODHD → Polygon → Yahoo → yfinance) + ISIN→ticker map
        │
        ▼
data/sqlite/eod_prices.sqlite
        │
        ▼
degiro_analytics/pipeline.py  →  metrics, optimization, regimes, charts
        │
        ▼
data/exports/  +  output/charts/
```

## Long-Running Cache

Price DB and DeGiro report CSVs are durable local state. Reuse unless `--refresh-degiro` or `--refresh-prices` is passed.

## Running

```powershell
# Full analysis (uses cache when available)
python scripts/run_degiro_analysis.py

# Force refresh
python scripts/run_degiro_analysis.py --refresh-degiro --refresh-prices

# Notebook
jupyter notebook "notebooks/20260528 - DeGiro portfolio analytics.ipynb"
```

## Notebook Naming

`YYYYMMDD - <topic>.ipynb`
