
# Mock Data for "Short Horizon Trap"
# Horizon = 3 hours
# Hour 0: DA=20, RT=50 (Trap: Small spike visible in RT but not DA)
# Hour 4: DA=100, RT=1000 (Prize: Huge spike later)

timestamps = pd.date_range(start='2025-01-01', periods=10, freq='h')
price_da = np.array([20, 20, 20, 20, 100, 20, 20, 20, 20, 20])
price_rt = np.array([50, 20, 20, 20, 1000, 20, 20, 20, 20, 20])

df = pd.DataFrame({
    'timestamp': timestamps,
    'price_mwh_da': price_da,
    'price_mwh_rt': price_rt
})
df['forecast_error'] = df['price_mwh_rt'] - df['price_mwh_da']

# Battery
specs = BatterySpecs(
    capacity_mwh=10,
    power_mw=10,
    efficiency=1.0, # Perfect efficiency for clarity
    max_soc=1.0,
    min_soc=0.0,
    initial_soc=10.0 # Start full to make decision "Discharge Now vs Later"
)

# Run Sweep
factors = [0.0, 1.0]
results = {}

print("Running MPC Sweep (Horizon=3h)...")
for f in factors:
    strategy = MPCStrategy(horizon_hours=3)
    battery = Battery(specs)
    battery.reset()
    # Force full charge
    battery.soc = 10.0
    
    total_rev = 0
    actions = []
    for i in range(len(df)):
        decision = strategy.decide(battery, i, df, f)
        actions.append(f"{decision.action}({decision.energy_mwh:.1f})")
        
        if decision.action == 'discharge':
            total_rev += decision.energy_mwh * decision.actual_price
            
    results[f] = total_rev
    print(f"Factor {f}: Revenue ${total_rev:,.2f} | Actions: {actions}")

