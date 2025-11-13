# ERCOT Battery Storage Revenue Opportunity Dashboard

**Zentus - Engie Urja AI Challenge 2025**

Interactive Streamlit dashboard demonstrating how improved renewable energy forecasting increases battery storage revenue in ERCOT markets.

## Quick Start

### 1. Install Dependencies

```bash
cd ~/ERCOT/ERCOT-Dashboard
pip install -r requirements.txt
```

### 2. Run the Dashboard

```bash
streamlit run app.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`

## Current Data

- **Date**: July 20, 2025 (24 hours)
- **Nodes**: 5 wind farm settlement points
  - BUFF_GAP_ALL
  - BAIRDWND_ALL
  - CEDROHI_CHW1
  - SWTWN4_WND45
  - WH_WIND_ALL
- **Records**: 768 hourly price points
- **Extreme Events**: 192 hours with >$10/MWh price spreads (25%)
- **Negative Prices**: 40 hours with negative RT prices

## Features

### Interactive Analysis
- Select different settlement points
- Adjust battery specifications (capacity, power, efficiency)
- Simulate forecast accuracy improvements (0-50%)

### Visualizations
- Real-time vs Day-ahead price comparison
- Battery state of charge across strategies
- Cumulative revenue comparison
- Sensitivity analysis of forecast improvement
- Forecast error distribution
- Price spread analysis

### Three Strategy Comparison
1. **Baseline**: Day-ahead forecast only
2. **Improved**: With forecast accuracy improvement
3. **Optimal**: Perfect foresight (theoretical maximum)

## Key Metrics

- **Revenue Opportunity**: Gap between baseline and optimal
- **Improvement Impact**: Revenue gain from better forecasting
- **Extreme Event Analysis**: Focus on high-volatility hours
- **Negative Price Events**: Unique arbitrage opportunities

## Data Sources

Price data processed from ERCOT shadow-monitor repository:
- `data/da_prices.csv` - Day-ahead market prices
- `data/rt_prices.csv` - Real-time market prices

Format: `timestamp, node, price_mwh`

## Project Structure

```
ERCOT-Dashboard/
├── app.py              # Main Streamlit application
├── data/              # Price data CSVs
│   ├── da_prices.csv
│   └── rt_prices.csv
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## Team

- **Juan Boullosa** - Dashboard development
- **Nicholas** - Data processing (ERCOT-shadow-monitor)
- **Aoife Henry** - Strategy and submission
- **Rafa Mudafort** - Database infrastructure
- **Ishaan Sood** - Market context

## Contact

**Zentus** - Intelligent Forecasting for Renewables
Stanford Doerr School of Sustainability Accelerator Fellow

Email: jmboullosa@zentus.io
Website: zentus.io

## Next Steps

1. Request additional historical data from Nicholas showing extreme events
2. Add hub-level prices (HB_NORTH, HB_HOUSTON) for broader analysis
3. Refine battery dispatch strategy based on market insights
4. Prepare 2-3 page analysis document for submission
5. Deploy dashboard for demo
