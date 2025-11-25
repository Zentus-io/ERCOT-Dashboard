import pandas as pd
import numpy as np
from core.battery.battery import BatterySpecs
from core.battery.simulator import BatterySimulator
from core.battery.strategies import MPCStrategy

def create_dummy_data(days=5):
    """Create dummy price data with a daily pattern."""
    timestamps = pd.date_range(start='2024-01-01', periods=24*days, freq='1h')
    
    # Create a daily price pattern: low at night, high in evening
    hours = timestamps.hour
    prices = 20 + 10 * np.sin((hours - 6) * np.pi / 12)  # Simple wave
    # Convert to numpy array to allow mutation if it's not already (it is, but let's be safe)
    prices = np.array(prices)
    prices[hours == 18] += 100  # Spike at 6 PM
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'price_mwh_da': prices,
        'price_mwh_rt': prices, # Perfect forecast for simplicity
        'forecast_error': 0.0
    })
    return df

def test_mpc_sensitivity():
    print("Running MPC Sensitivity Test...")
    
    # Setup
    specs = BatterySpecs(capacity_mwh=10, power_mw=5, efficiency=0.9)
    simulator = BatterySimulator(specs)
    data = create_dummy_data(days=3)
    
    horizons = [2, 4, 12, 24]
    revenues = []
    
    for h in horizons:
        print(f"Testing horizon: {h} hours...")
        strategy = MPCStrategy(horizon_hours=h)
        result = simulator.run(data, strategy, improvement_factor=1.0)
        revenues.append(result.total_revenue)
        print(f"  Revenue: ${result.total_revenue:,.2f}")
        
    # Check if revenue increases with horizon (at least initially)
    # 2h should be worse than 12h because it can't see the evening spike from the morning
    
    print("\nResults:")
    for h, rev in zip(horizons, revenues):
        print(f"Horizon {h}h: ${rev:,.2f}")
        
    if revenues[0] < revenues[2]:
        print("\nSUCCESS: Short horizon (2h) performs worse than long horizon (12h). Sensitivity logic is working.")
    else:
        print("\nWARNING: Short horizon performed same/better. Check logic.")

if __name__ == "__main__":
    test_mpc_sensitivity()
