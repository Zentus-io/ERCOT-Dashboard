# ERCOT Dashboard Scripts

This directory contains utility scripts for data fetching, database management, and system setup.

## 🚀 Core Data Fetching

### `fetch_ercot_generation.py`

**Purpose:** Fetches hourly **Wind and Solar generation actuals** from ERCOT (via GridStatus) and maps them to Engie's asset settlement points.
**Usage:**

```bash
# Fetch last 7 days (default)
python fetch_ercot_generation.py

# Fetch specific range
python fetch_ercot_generation.py --start 2025-01-01 --end 2025-01-31

# Fetch last 30 days
python fetch_ercot_generation.py --days 30
```

**Key Features:**

- Uses `gridstatus` library (v0.33.0+).
- Maps ERCOT regions (e.g., "CenterWest") to Asset Zones (e.g., "West").
- Upserts data to `ercot_generation` table in Supabase.

### `fetch_ercot_data.py`

**Purpose:** Fetches **DAM (Day-Ahead Market) and RTM (Real-Time Market) prices** for settlement points.
**Usage:**

```bash
# Fetch prices for a specific date range
python fetch_ercot_data.py --start_date 2025-01-01 --end_date 2025-01-07
```

### `fetch_ercot_bulk_reports.py`

**Purpose:** Downloads raw ZIP/CSV reports directly from the ERCOT Public API. Useful for bulk historical archives or reports not supported by `gridstatus`.
**Usage:**

```bash
python fetch_ercot_bulk_reports.py --report RTM --start 2025-01-01
python fetch_ercot_bulk_reports.py --report WIND_GEN_GEO SOLAR_GEN_GEO --start 2025-01-01
```

**Available Reports:**

- `RTM` / `RTM_HISTORICAL`
- `DAM` / `DAM_AS`
- `WIND_GEN_LZ` (Hourly Load Zone) / `WIND_GEN_GEO` (Hourly Geo) / `WIND_GEN_5MIN` (5-min Actuals)
- `SOLAR_GEN_LZ` (Hourly Load Zone) / `SOLAR_GEN_GEO` (Hourly Geo) / `SOLAR_GEN_5MIN` (5-min Actuals)

## 🛠️ Database Setup & Migration

### `setup_supabase_schema.sql`

**Purpose:** Defines the core database schema for `ercot_prices` and `engie_storage_assets`. Run this in the Supabase SQL Editor to initialize the project.

### `setup_generation_schema.sql`

**Purpose:** Defines the schema for the `ercot_generation` table and its RLS policies. Run this to enable Wind/Solar data storage.

### `migrate_existing_data.py`

**Purpose:** One-time script to upload local CSV files (e.g., historical prices) to Supabase.

### `upload_engie_assets.py`

**Purpose:** Uploads the list of Engie storage assets and their metadata to the `engie_storage_assets` table.

## 🧹 Data Management

### `delete_prices_range.py`

**Purpose:** Deletes price data (DAM/RTM) from the `ercot_prices` table for a specific date range. Useful for cleaning up partial fetches.
**Usage:**

```bash
python delete_prices_range.py --start 2025-11-20 --end 2025-11-26
```

### `reset_generation_data.py`

**Purpose:** Deletes **ALL** data from the `ercot_generation` table. Use with caution.
**Usage:**

```bash
python reset_generation_data.py
```

## 🧪 Utilities

### `test_database_connection.py`

**Purpose:** Verifies connectivity to Supabase and checks if tables exist.
**Usage:**

```bash
python test_database_connection.py
```

### `preprocess_eia860_battery_data.py`

**Purpose:** Processes raw EIA-860 data files to extract battery storage metadata.

## 📚 Glossary & Acronyms

| Acronym | Definition | Description |
| :--- | :--- | :--- |
| **STWPF** | Short-Term Wind Power Forecast | ERCOT's official 50% probability of exceedance forecast for wind generation. |
| **STPPF** | Short-Term PhotoVoltaic Power Forecast | ERCOT's official forecast for solar generation. |
| **WGRPP** | Wind Generation Resource Power Potential | The theoretical maximum power a wind farm could produce given current weather conditions (ignoring curtailment). |
| **PVGRPP** | PhotoVoltaic Generation Resource Power Potential | The theoretical maximum power a solar farm could produce given current weather conditions. |
| **HSL** | High Sustainable Limit | The maximum power output a resource can currently sustain. Used by ERCOT for dispatch. |
| **COP** | Current Operating Plan | The resource's planned status and capacity as reported to ERCOT. |
| **LZ** | Load Zone | Aggregated zones (West, North, South, Houston) used primarily for pricing load. |
| **Geo** | Geographical Region | Weather-based regions (Panhandle, Coastal, etc.) used for renewable forecasting. |
