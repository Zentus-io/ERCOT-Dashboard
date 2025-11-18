"""
Application Settings and Constants
Zentus - ERCOT Battery Revenue Dashboard
"""

# App metadata
APP_TITLE = "Zentus - ERCOT Revenue Opportunity"
APP_SUBTITLE = "Demonstrating Value Through Intelligent Forecasting"
APP_ICON = "⚡"

# Color scheme
COLORS = {
    'primary': '#0A5F7A',
    'secondary': '#1E3A5F',
    'accent': '#4A9FB8',
    'success': '#28A745',
    'warning': '#FFC107',
    'danger': '#DC3545',
    'gray': '#6C757D',
    'light_gray': '#6B7280',
    'background': '#f0f8fa',
}

# Chart colors
CHART_COLORS = {
    'rt_price': '#0A5F7A',
    'da_price': '#FF6B35',
    'baseline': '#DC3545',
    'improved': '#FFC107',
    'optimal': '#28A745',
    'charge': {'light': 'rgba(40, 167, 69, 0.3)', 'dark': '#28A745'},
    'discharge': {'light': 'rgba(220, 53, 69, 0.3)', 'dark': '#DC3545'},
    'hold': {'light': '#6C757D', 'dark': '#6C757D'},
}

# Battery defaults
DEFAULT_BATTERY = {
    'capacity_mwh': 100,
    'power_mw': 50,
    'efficiency': 0.9,
    'min_soc': 0.05,
    'max_soc': 0.95,
    'initial_soc': 0.5,
}

# Strategy defaults
DEFAULT_STRATEGY = {
    'type': 'Threshold-Based',
    'charge_percentile': 0.25,
    'discharge_percentile': 0.75,
    'window_hours': 6,
}

# Simulation defaults
DEFAULT_FORECAST_IMPROVEMENT = 10  # %

# Price thresholds
EXTREME_EVENT_THRESHOLD = 10  # $/MWh spread
MINIMUM_ARBITRAGE_SPREAD = 5  # $/MWh

# EIA Battery presets (based on real Texas data)
BATTERY_PRESETS = {
    'Small': {
        'name': 'Small (TX Median: 10 MW / 17 MWh)',
        'capacity_mwh': 17,
        'power_mw': 10,
    },
    'Medium': {
        'name': 'Medium (TX Mean: 59 MW / 85 MWh)',
        'capacity_mwh': 85,
        'power_mw': 59,
    },
    'Large': {
        'name': 'Large (TX 90th percentile)',
        'capacity_mwh': None,  # Calculated from EIA data
        'power_mw': None,
    },
    'Very Large': {
        'name': 'Very Large (TX Max: 300 MW)',
        'capacity_mwh': 600,
        'power_mw': 300,
    },
}

# Cache settings
CACHE_TTL = 3600  # 1 hour in seconds

# File paths
DATA_NOTE = """
**MVP Demo:** Currently showing July 20, 2025 data from ERCOT wind resources.
This single-day snapshot demonstrates the revenue opportunity concept.
Additional historical data with extreme price events is being processed.
"""
