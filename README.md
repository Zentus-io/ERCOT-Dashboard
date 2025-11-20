# ERCOT Battery Storage Revenue Opportunity Dashboard

## Zentus - Engie Urja AI Challenge 2025

Interactive Streamlit dashboard demonstrating how improved renewable energy forecasting increases battery storage revenue in ERCOT markets. Features a modular OOP architecture with Supabase database integration for scalable historical data analysis.

---

## 🚀 Quick Start

1. **Clone the repository**
2. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the dashboard**:

    ```bash
    streamlit run Home.py
    ```

If you want to use the Supabase database integration:

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your credentials:
# - Supabase URL and API key
# - ERCOT API credentials (optional)
```

### 3. Run the Dashboard

```bash
streamlit run app.py
```

The dashboard will open automatically at `http://localhost:8501`

---

## 📊 Features

### Multi-Page Analysis Suite

The dashboard provides 7 specialized analysis pages:

1. **🏠 Overview** - Strategy performance comparison and key revenue metrics
2. **📈 Price Analysis** - Price dynamics, forecast errors, and extreme events
3. **🔋 Operations** - State of charge tracking and dispatch action distribution
4. **💰 Revenue** - Cumulative revenue tracking and pricing analysis
5. **🎯 Opportunity** - Sensitivity analysis across forecast improvement levels
6. **📊 Timeline** - Gantt-style dispatch visualization showing charge/discharge patterns
7. **⚙️ Optimization** - Deep-dive into strategy decision-making logic

### Interactive Configuration

- **Settlement Point Selection**: Analyze different ERCOT nodes
- **Battery Specifications**: Adjust capacity (10-600 MWh), power (5-300 MW), efficiency
- **EIA-860 Presets**: Use real Texas battery system configurations (Small/Medium/Large)
- **Strategy Selection**:
  - Threshold-Based (percentile-driven charge/discharge)
  - Rolling Window Optimization (lookahead planning)
- **Forecast Improvement**: Simulate 0-100% accuracy improvements

### Three-Scenario Comparison

Every analysis compares:

1. **Baseline**: Day-ahead forecast only (current capability)
2. **Improved**: With Zentus forecast enhancement (+X%)
3. **Optimal**: Perfect foresight (theoretical maximum revenue)

---

## 🗄️ Data Architecture

### Dual Data Sources

The dashboard supports two data loading modes:

#### CSV Mode (Default)

- **Location**: `data/da_prices.csv`, `data/rt_prices.csv`
- **Use case**: Local development, offline analysis
- **Data**: Single-day snapshot (July 20, 2025)
- **Nodes**: 5 wind farm settlement points

#### Database Mode (Supabase)

- **Location**: Cloud PostgreSQL database
- **Use case**: Production, historical analysis, team collaboration
- **Data**: Years of ERCOT market data
- **Nodes**: All ERCOT settlement points
- **Features**: Date range selection, automatic updates, query optimization

The system automatically detects Supabase credentials and uses the database if configured, otherwise falls back to CSV.

---

## 🏗️ Project Structure

```
ERCOT-Dashboard/
├── app.py                          # Streamlit entry point (115 lines - refactored!)
│
├── pages/                          # Multi-page analysis views
│   ├── 1_🏠_Overview.py           # Strategy performance comparison
│   ├── 2_📈_Price_Analysis.py     # Price dynamics and forecast errors
│   ├── 3_🔋_Operations.py         # SOC and dispatch analysis
│   ├── 4_💰_Revenue.py            # Revenue tracking over time
│   ├── 5_🎯_Opportunity.py        # Sensitivity analysis
│   ├── 6_📊_Timeline.py           # Dispatch timeline visualization
│   └── 7_⚙️_Optimization.py      # Strategy deep-dive
│
├── core/                           # Business logic (OOP design)
│   ├── battery/
│   │   ├── battery.py             # BatterySpecs & Battery classes
│   │   ├── simulator.py           # BatterySimulator orchestration
│   │   └── strategies.py          # DispatchStrategy implementations
│   ├── data/
│   │   └── loaders.py             # DataLoader & SupabaseDataLoader
│   └── analytics/                 # Analysis utilities
│
├── ui/                             # User interface components
│   ├── components/
│   │   ├── sidebar.py             # Shared configuration sidebar
│   │   └── header.py              # Page header with branding
│   └── styles/
│       └── custom_css.py          # Custom CSS styling
│
├── config/                         # Configuration
│   ├── settings.py                # Constants, colors, database config
│   └── page_config.py             # Streamlit page settings
│
├── utils/                          # Utilities
│   └── state.py                   # Session state management (AppState)
│
├── scripts/                        # Database & data management
│   ├── setup_supabase_schema.sql  # Database DDL (tables, indexes, views)
│   ├── create_supabase_tables.py  # Schema setup helper
│   ├── fetch_ercot_data.py        # Fetch new data from ERCOT API
│   ├── migrate_existing_data.py   # One-time CSV → DB migration
│   └── test_database_connection.py # Database connectivity test
│
├── data/                           # CSV data (fallback/local dev)
│   ├── da_prices.csv              # Day-ahead market prices
│   ├── rt_prices.csv              # Real-time market prices
│   └── processed/
│       └── ercot_batteries.parquet # EIA-860 battery market data
│
├── .env.example                    # Environment variable template
├── .gitignore                      # Git ignore rules (includes .env)
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

### Architecture Highlights

- **Object-Oriented Design**: Dataclasses for immutable state, Strategy pattern for dispatch algorithms
- **Separation of Concerns**: Business logic (core/) separate from UI (ui/)
- **Caching**: Streamlit `@st.cache_data` for expensive operations
- **Modularity**: 1,919-line monolith refactored into ~10 focused modules
- **Database Abstraction**: Unified `load_data()` function supports multiple sources

---

## 🔧 Database Setup (Optional)

To use the Supabase database integration for historical data:

### Step 1: Set Up Supabase Project

1. Create account at [https://supabase.com](https://supabase.com)
2. Create new project
3. Copy Project URL and API Key (anon public)

### Step 2: Configure Environment

```bash
cp .env.example .env
# Edit .env with your credentials
```

### Step 3: Create Database Schema

1. Open Supabase SQL Editor in your project dashboard
2. Copy contents of `scripts/setup_supabase_schema.sql`
3. Run the SQL to create tables, indexes, and views

Alternatively:

```bash
python scripts/create_supabase_tables.py
# Follow the instructions to manually execute SQL
```

### Step 4: Migrate Existing Data

```bash
# Import dashboard CSV data
python scripts/migrate_existing_data.py

# Optionally include live data from ../ERCOT-Live-Data/
python scripts/migrate_existing_data.py --include-live-data
```

### Step 5: Verify Setup

```bash
python scripts/test_database_connection.py
```

### Step 6: Fetch New Data (Optional)

To populate with fresh ERCOT market data:

```bash
# Requires ERCOT API credentials in .env
# Fetch last 7 days
python scripts/fetch_ercot_data.py

# Fetch specific date range
python scripts/fetch_ercot_data.py --start 2025-01-01 --end 2025-01-31 --refresh-view
```

The dashboard will automatically use the database when credentials are configured!

---

## 📈 Current Data

### CSV Mode (Default)

- **Date**: July 20, 2025 (24 hours)
- **Nodes**: 5 wind farm settlement points
  - BUFF_GAP_ALL
  - BAIRDWND_ALL
  - CEDROHI_CHW1
  - SWTWN4_WND45
  - WH_WIND_ALL
- **Records**: 768 hourly price points
- **Extreme Events**: 192 hours with >$10/MWh spreads (25%)
- **Negative Prices**: 40 hours with negative RT prices

### Database Mode (After Setup)

- **Date Range**: Configurable (depends on data ingestion)
- **Nodes**: All ERCOT resource nodes
- **Records**: Potentially millions (DAM hourly + RTM 15-min)
- **Update Frequency**: As configured (manual or scheduled)

---

## 🎓 Key Concepts

### Revenue Opportunity

**Revenue Opportunity = Optimal Strategy Revenue - Baseline Strategy Revenue**

This metric quantifies the value of improved forecasting:

- **Baseline**: What's possible with DA forecasts alone
- **Improved**: What Zentus's enhanced forecasting enables
- **Optimal**: Theoretical maximum (perfect foresight benchmark)

### Extreme Events

Price spreads >$10/MWh between DA and RT markets. These high-volatility periods create the biggest arbitrage opportunities and account for most of the revenue opportunity.

### Forecast Improvement Factor

**0%** = Use only day-ahead forecasts (baseline)
**50%** = Correct 50% of the forecast error
**100%** = Perfect real-time price knowledge (optimal)

The dashboard shows how revenue scales with forecast accuracy improvements.

### Dispatch Strategies

**Threshold-Based**:

- Charge when price < 25th percentile
- Discharge when price > 75th percentile
- Fast, simple, works well with better forecasts

**Rolling Window Optimization**:

- Looks ahead N hours
- Solves for optimal charge/discharge pattern
- More sophisticated, benefits more from accuracy

---

## 📊 Visualizations

- **Price Charts**: Interactive Plotly charts with zoom/pan
- **State of Charge**: Real-time SOC tracking across scenarios
- **Revenue Curves**: Cumulative revenue comparison
- **Sensitivity Analysis**: Revenue vs forecast improvement curves
- **Dispatch Timeline**: Gantt-style charge/discharge visualization
- **Distribution Analysis**: Histogram of price forecasts vs actuals

All charts use Zentus brand colors and support dark/light mode.

---

## 🔬 Technical Details

### Simulation Engine

The `BatterySimulator` class orchestrates:

1. Price data ingestion (CSV or database)
2. Strategy selection (Threshold or Rolling Window)
3. Timestep-by-timestep dispatch decisions
4. SOC tracking with efficiency losses
5. Revenue calculation (actual RT prices)
6. Result aggregation

### Forecast Improvement Model

```python
improved_price = da_price + (improvement_factor * forecast_error)
# where forecast_error = rt_price - da_price
```

This simulates having a better forecast that captures X% of the DA-to-RT price movement.

### Database Schema

- **ercot_prices**: Raw price data (timestamp, location, market, price)
- **ercot_prices_merged**: Materialized view (pre-joined DAM + RTM with metrics)
- **eia_batteries**: Texas battery market reference data
- **Indexes**: Optimized for location + market + timestamp queries

---

## 🧪 Testing

```bash
# Test database connection
python scripts/test_database_connection.py

# Run dashboard in test mode
streamlit run app.py

# Check dependencies
pip check
```

---

## 👥 Team

**Zentus** - Intelligent Forecasting for Renewables

- **Juan Boullosa** - Dashboard development & OOP architecture
- **N** - Data processing (ERCOT-shadow-monitor)
- **Aoife Henry** - Strategy and submission
- **Rafa Mudafort** - Database infrastructure
- **Ishaan Sood** - Market context

---

## 📧 Contact

**Zentus**
Stanford Doerr School of Sustainability Accelerator Fellow

**Email**: <jmboullosa@zentus.io>
**Website**: [https://zentus.io](https://zentus.io)

---

## 📝 License & Data Sources

### ERCOT Market Data

- **Source**: Electric Reliability Council of Texas (ERCOT)
- **Access**: Public API via [gridstatus](https://github.com/kmax12/gridstatus) library
- **Markets**: Day-Ahead Market (DAM), Real-Time Market (RTM)
- **License**: ERCOT data usage subject to ERCOT protocols

### EIA Battery Data

- **Source**: U.S. Energy Information Administration (EIA-860)
- **Dataset**: Form EIA-860 Energy Storage Inventory (2024)
- **Scope**: 136 operational battery systems in Texas (ERCOT)
- **License**: Public domain

### Dashboard Code

- **Author**: Zentus
- **Purpose**: Engie Urja AI Challenge 2025 submission
- **Framework**: Streamlit 1.28+
- **Database**: Supabase (PostgreSQL)

---

## 🚀 Future Enhancements

Potential additions (not implemented in MVP):

- **Real-time data streaming**: WebSocket connection to ERCOT API
- **Advanced strategies**: Machine learning-based dispatch
- **Multi-market optimization**: Combine arbitrage + ancillary services
- **Portfolio analysis**: Multi-battery fleet optimization
- **Weather integration**: Renewable generation forecasts
- **Cost modeling**: Include degradation, O&M, demand charges
- **Export functionality**: CSV/PDF report generation

---

## 📚 Additional Documentation

- **Supabase Setup**: See `scripts/create_supabase_tables.py` docstrings
- **Data Schema**: See `scripts/setup_supabase_schema.sql` comments
- **API Usage**: See `scripts/fetch_ercot_data.py` for ERCOT API examples
- **Strategy Details**: See `core/battery/strategies.py` docstrings

---

**Built with ❤️ by Zentus - Making renewable energy more predictable and profitable**
