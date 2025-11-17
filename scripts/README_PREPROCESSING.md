# EIA-860 Battery Data Preprocessing

## Overview

This directory contains scripts to preprocess EIA Form 860 Energy Storage data for use in the ERCOT Battery Revenue Dashboard.

## Files

- **`preprocess_eia860_battery_data.py`** - Main preprocessing script
- **`README_PREPROCESSING.md`** - This documentation file

---

## Quick Start

### 1. Install Dependencies

Ensure you have the required Python packages:

```bash
# Activate your conda/mamba environment
mamba activate ercot

# Install dependencies (if not already installed)
pip install polars pandas openpyxl pyarrow
```

### 2. Run the Script

From the project root directory:

```bash
cd ERCOT-Dashboard
python scripts/preprocess_eia860_battery_data.py
```

### 3. Output

The script will create:

```
data/processed/ercot_batteries.parquet
```

---

## What the Script Does

### Pipeline Steps

1. **Load Excel File** → Reads `data/eia8602024/3_4_Energy_Storage_Y2024.xlsx` (Operable sheet)
2. **Convert to Polars** → Converts to Polars DataFrame for fast processing
3. **Filter Texas Batteries** → Extracts only batteries in Texas (ERCOT region)
4. **Select Important Columns** → Keeps 30+ relevant columns, renames for clarity
5. **Calculate Derived Metrics** → Adds duration, energy-to-power ratio, primary use case
6. **Save as Parquet** → Exports optimized file with ZSTD compression

### Data Transformations

- **Filtering**: Only Texas (State = 'TX') batteries
- **Sorting**: By nameplate power capacity (largest first)
- **Derived Columns**:
  - `duration_hours` = energy (MWh) / power (MW)
  - `e_to_p_ratio` = energy-to-power ratio
  - `primary_use_case` = Categorized primary function (Arbitrage, Frequency Regulation, etc.)

### Optimization

- **Format**: Parquet with ZSTD compression
- **Statistics**: Enabled for query optimization
- **Size Reduction**: Typically ~70-80% smaller than Excel
- **Loading Speed**: 10-50x faster than Excel

---

## Output File Structure

### File Location

```
data/processed/ercot_batteries.parquet
```

### Column Reference (31 columns)

#### **Identification (5 columns)**
| Column | Type | Description |
|--------|------|-------------|
| `plant_code` | Int | EIA plant code (unique identifier) |
| `plant_name` | String | Power plant name |
| `state` | String | State (always 'TX' for this dataset) |
| `county` | String | County name |
| `generator_id` | String | Generator ID within plant |

#### **Technical Specifications (9 columns)**
| Column | Type | Description |
|--------|------|-------------|
| `technology` | String | Generation technology type |
| `prime_mover` | String | Energy storage prime mover code |
| `storage_technology` | String | Storage technology (e.g., Lithium-ion, Flywheel) |
| `status` | String | Operational status |
| `nameplate_power_mw` | Float | Maximum power output (MW) |
| `summer_power_mw` | Float | Summer capacity (MW) |
| `winter_power_mw` | Float | Winter capacity (MW) |
| `nameplate_energy_mwh` | Float | Total energy storage capacity (MWh) |
| `max_charge_mw` | Float | Maximum charging rate (MW) |
| `max_discharge_mw` | Float | Maximum discharge rate (MW) |

#### **Operational Info (4 columns)**
| Column | Type | Description |
|--------|------|-------------|
| `operating_month` | Int | Month system became operational |
| `operating_year` | Int | Year system became operational |
| `sector_name` | String | Sector name (e.g., IPP Non-CHP, Electric Utility) |
| `sector_code` | Int | EIA sector code |

#### **Use Cases (8 columns)**
| Column | Type | Description |
|--------|------|-------------|
| `use_arbitrage` | String | 'Y' if used for price arbitrage |
| `use_frequency_regulation` | String | 'Y' if used for frequency regulation |
| `use_load_following` | String | 'Y' if used for load following |
| `use_ramping_reserve` | String | 'Y' if used for ramping/spinning reserve |
| `use_renewable_firming` | String | 'Y' if used for renewable firming |
| `use_td_deferral` | String | 'Y' if used for T&D deferral |
| `use_peak_shaving` | String | 'Y' if used for peak shaving |
| `use_load_management` | String | 'Y' if used for load management |

#### **Coupling & Configuration (3 columns)**
| Column | Type | Description |
|--------|------|-------------|
| `ac_coupled` | String | 'Y' if AC-coupled |
| `dc_coupled` | String | 'Y' if DC-coupled |
| `independent` | String | 'Y' if independent system |

#### **Derived Metrics (3 columns)**
| Column | Type | Description |
|--------|------|-------------|
| `duration_hours` | Float | Storage duration in hours (MWh / MW) |
| `e_to_p_ratio` | Float | Energy-to-power ratio (same as duration) |
| `primary_use_case` | String | Primary use case category |

---

## Loading Processed Data

### In Python (Polars)

```python
import polars as pl

# Fast loading (recommended)
df = pl.read_parquet('data/processed/ercot_batteries.parquet')

# Example queries
texas_arbitrage = df.filter(pl.col('use_arbitrage') == 'Y')
large_systems = df.filter(pl.col('nameplate_power_mw') > 100)
short_duration = df.filter(pl.col('duration_hours') < 2)
```

### In Python (Pandas)

```python
import pandas as pd

# Load with pandas (slower but compatible)
df = pd.read_parquet('data/processed/ercot_batteries.parquet')

# Example queries
arbitrage_batteries = df[df['use_arbitrage'] == 'Y']
median_capacity = df['nameplate_power_mw'].median()
```

### In Streamlit Dashboard

```python
import polars as pl
import streamlit as st

@st.cache_data
def load_battery_market_data():
    """Load preprocessed ERCOT battery data."""
    return pl.read_parquet('data/processed/ercot_batteries.parquet')

# Use in dashboard
battery_data = load_battery_market_data()
```

---

## Performance Comparison

### File Size

| Format | Size | Compression |
|--------|------|-------------|
| Original Excel (.xlsx) | ~1.2 MB | - |
| Parquet (ZSTD) | ~0.3 MB | 75% reduction |

### Loading Speed (136 Texas batteries)

| Method | Time | Speedup |
|--------|------|---------|
| `pd.read_excel()` | ~500-800ms | 1x |
| `pd.read_parquet()` | ~50-80ms | **8-10x faster** |
| `pl.read_parquet()` | ~15-30ms | **20-30x faster** |

### Memory Usage

| Method | Memory | Efficiency |
|--------|--------|------------|
| Pandas DataFrame | ~2-3 MB | 1x |
| Polars DataFrame | ~1-1.5 MB | **2x more efficient** |

---

## Data Quality Checks

The preprocessing script includes automatic quality checks:

✅ **Filters valid data**: Only operational Texas batteries
✅ **Validates capacity values**: Non-null nameplate capacity required
✅ **Calculates duration**: Derived metric for quick filtering
✅ **Sorts by size**: Largest systems first for easy inspection
✅ **Preserves all use cases**: Binary flags for multiple applications

---

## Updating the Data

When EIA releases new Form 860 data:

1. Download the new `3_4_Energy_Storage_Y2024.xlsx` file
2. Place in `data/eia8602024/` directory
3. Run the preprocessing script:
   ```bash
   python scripts/preprocess_eia860_battery_data.py
   ```
4. The Parquet file will be regenerated with new data
5. Restart the Streamlit dashboard to use updated data

---

## Troubleshooting

### Error: `ModuleNotFoundError: No module named 'polars'`

**Solution**: Install Polars
```bash
pip install polars
```

### Error: `FileNotFoundError: data/eia8602024/3_4_Energy_Storage_Y2024.xlsx`

**Solution**: Ensure the Excel file exists in the correct location
```bash
ls data/eia8602024/3_4_Energy_Storage_Y2024.xlsx
```

### Error: `openpyxl` not installed

**Solution**: Install openpyxl
```bash
pip install openpyxl
```

### Slow performance

**Solution**: Ensure PyArrow is installed for fast Parquet I/O
```bash
pip install pyarrow
```

---

## Summary Statistics (Example Output)

```
================================================================================
DATA SUMMARY
================================================================================
Total Texas batteries:        136
Total installed power:        8,059.5 MW
Total installed energy:       11,505.0 MWh
Average system size:          59.3 MW
Median system size:           10.0 MW
Average duration:             1.43 hours
Median duration:              1.00 hours

Primary Use Cases:
  Arbitrage                      74 systems ( 54.4%)
  Ramping Reserve                38 systems ( 27.9%)
  Frequency Regulation           20 systems ( 14.7%)
  Other                           4 systems (  2.9%)

Storage Technologies:
  Lithium-ion                   132 systems ( 97.1%)
  Flywheel                        2 systems (  1.5%)
  Other                           2 systems (  1.5%)
================================================================================
```

---

## Notes

- **Data Source**: U.S. Energy Information Administration (EIA) Form EIA-860
- **Update Frequency**: EIA releases annual updates (typically October/November)
- **Geographic Scope**: Texas only (ERCOT region)
- **Technology Filter**: Battery storage systems (excludes other generation types)
- **Status Filter**: Operable systems only (excludes proposed/retired)

---

## Questions?

For issues or questions about the preprocessing script, contact the Zentus team or refer to the main dashboard documentation.
