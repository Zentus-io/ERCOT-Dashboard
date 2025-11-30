import sys
import pandas as pd
from gridstatus import Ercot
from datetime import date

def main():
    iso = Ercot()
    d = date(2025, 11, 15)
    print(f"Fetching Solar for {d}...")
    
    try:
        df_solar = pd.DataFrame()
        if hasattr(iso, 'get_solar_actual_and_forecast_by_geographical_region_hourly'):
            print("Using get_solar_actual_and_forecast_by_geographical_region_hourly")
            df_solar = iso.get_solar_actual_and_forecast_by_geographical_region_hourly(date=d)
        elif hasattr(iso, 'get_hourly_solar_report'):
            print("Using get_hourly_solar_report")
            df_solar = iso.get_hourly_solar_report(date=d)
            
        if df_solar.empty:
            print("❌ Solar DF is empty")
        else:
            print("✅ Solar DF fetched")
            print("Columns:", df_solar.columns.tolist())
            print("First 5 rows:")
            print(df_solar.head())
            
            # Check for GEN columns
            gen_cols = [c for c in df_solar.columns if c.startswith('GEN ') and 'SYSTEM' not in c]
            print("GEN Columns:", gen_cols)
            
            # Check regions
            regions = [c.replace('GEN ', '') for c in gen_cols]
            print("Regions found:", regions)
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
