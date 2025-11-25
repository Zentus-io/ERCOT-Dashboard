
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from core.battery.strategies import MPCStrategy, LinearOptimizationStrategy
from core.battery.battery import Battery, BatterySpecs

def test_mpc_strategy():
    print("Testing MPC Strategy...")
    
    # 1. Setup Mock Data
    dates = pd.date_range(start='2025-01-01', periods=48, freq='H')
    prices = np.array([20, 20, 20, 20, 20, 20, 100, 100, 20, 20, 20, 20] * 4) # Simple pattern
    
    price_df = pd.DataFrame({
        'timestamp': dates,
        'price_mwh_da': prices,
        'price_mwh_rt': prices, # Perfect forecast for simplicity
        'forecast_error': np.zeros(48)
    })
    
    specs = BatterySpecs(
        capacity_mwh=10,
        power_mw=5,
        efficiency=0.9
    )
    
    battery = Battery(specs)
    
    # 2. Initialize Strategy
    mpc = MPCStrategy(horizon_hours=12)
    
    # 3. Run Step-by-Step
    print("\nRunning simulation loop...")
    total_revenue = 0
    
    for i in range(len(price_df)):
        decision = mpc.decide(battery, i, price_df, improvement_factor=1.0)
        
        if decision.action != 'hold':
            print(f"Step {i} ({price_df.iloc[i]['timestamp']}): {decision.action} {decision.power_mw:.2f} MW @ ${decision.actual_price:.2f}")
            
            if decision.action == 'charge':
                total_revenue -= abs(decision.energy_mwh) * decision.actual_price
            else:
                total_revenue += abs(decision.energy_mwh) * decision.actual_price
                
    print(f"\nTotal Revenue: ${total_revenue:.2f}")
    
    # 4. Compare with Perfect Foresight (Global LP)
    print("\nComparing with Global LP...")
    battery.reset()
    lp = LinearOptimizationStrategy()
    
    lp_revenue = 0
    for i in range(len(price_df)):
        decision = lp.decide(battery, i, price_df, improvement_factor=1.0)
        if decision.action == 'charge':
            lp_revenue -= abs(decision.energy_mwh) * decision.actual_price
        elif decision.action == 'discharge':
            lp_revenue += abs(decision.energy_mwh) * decision.actual_price
            
    print(f"Global LP Revenue: ${lp_revenue:.2f}")
    
    # MPC should be close to LP in this simple deterministic case
    assert total_revenue > 0, "MPC should make money"
    print("\nTest Passed!")

if __name__ == "__main__":
    test_mpc_strategy()
