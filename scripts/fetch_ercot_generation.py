"""ERCOT Generation Data Fetcher (Supabase)
Zentus - ERCOT Battery Revenue Dashboard

Purpose:
    Fetches hourly **Wind and Solar generation actuals** from ERCOT (via `gridstatus`)
    and maps them to Engie's asset settlement points based on their region/zone.
    Data is stored in the Supabase `ercot_generation` table.

Usage:
    python scripts/fetch_ercot_generation.py --days 7
    python scripts/fetch_ercot_generation.py --help

Arguments:
    --start (str): Start date (YYYY-MM-DD).
    --end (str): End date (YYYY-MM-DD).
    --days (int): Number of past days to fetch if no dates provided. Default: 7.

Examples:
    # Fetch data for the last 7 days
    python scripts/fetch_ercot_generation.py

    # Fetch data for a specific range
    python scripts/fetch_ercot_generation.py --start 2025-11-13 --end 2025-11-26

    # Fetch data for the last 30 days
    python scripts/fetch_ercot_generation.py --days 30
"""

import argparse
import datetime
import os
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from supabase import Client, create_client
from tqdm import tqdm

# Try to import gridstatus, handle if missing (though required for this script)
try:
    import gridstatus
    from gridstatus import Ercot
    GRIDSTATUS_AVAILABLE = True
except ImportError:
    GRIDSTATUS_AVAILABLE = False
    print("⚠️  WARNING: gridstatus library not found. Please install it: pip install gridstatus")

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# ============================================================================
# CONFIGURATION
# ============================================================================

BATCH_SIZE = 500
MAX_RETRIES = 3


# Mapping from ERCOT Load Zones to Report Column Suffixes
# Solar Report Columns: GEN CenterWest, GEN NorthWest, GEN FarWest, GEN FarEast, GEN SouthEast, GEN CenterEast
# Wind Report Columns: GEN LZ SOUTH HOUSTON, GEN LZ WEST, GEN LZ NORTH

ZONE_MAPPING = {
    # Load Zone -> (Solar Suffix, Wind Suffix)
    'Houston': ('SouthEast', 'LZ SOUTH HOUSTON'),
    'South': ('SouthEast', 'LZ SOUTH HOUSTON'),
    'North': ('NorthWest', 'LZ NORTH'),
    'North/Austin': ('NorthWest', 'LZ NORTH'),
    'West': ('CenterWest', 'LZ WEST'), 
    'FarWest': ('FarWest', 'LZ WEST'),
    'Panhandle': ('NorthWest', 'LZ NORTH'),
    'Coastal': ('SouthEast', 'LZ SOUTH HOUSTON'),
    'East': ('FarEast', 'LZ SOUTH HOUSTON'),
    'SouthCentral': ('CenterEast', 'LZ SOUTH HOUSTON'),
    'Unknown': ('CenterWest', 'LZ WEST'),
}

DEFAULT_MAPPING = ('CenterWest', 'LZ WEST')

def get_zone_mapping(zone: str) -> tuple[str, str]:
    """Returns (SolarSuffix, WindSuffix) for a given Load Zone."""
    if not zone:
        return DEFAULT_MAPPING
    
    z = zone.replace(' ', '').lower()
    
    # Direct lookup first
    if zone in ZONE_MAPPING:
        return ZONE_MAPPING[zone]
        
    # Fuzzy lookup
    for key, val in ZONE_MAPPING.items():
        if key.lower() in z:
            return val
            
    print(f"    ⚠️  Warning: No mapping found for zone '{zone}'. Using default.")
    return DEFAULT_MAPPING

def fetch_generation_data(iso: Any, date_to_fetch: date) -> pd.DataFrame:
    """Fetches Solar and Wind regional actuals for the given date."""
    print(f"DEBUG: Fetching for {date_to_fetch}")
    dfs = []
    
    # 1. Solar (Hourly Actuals + Forecasts)
    try:
        df_solar = pd.DataFrame()
        if hasattr(iso, 'get_solar_actual_and_forecast_by_geographical_region_hourly'):
            df_solar = iso.get_solar_actual_and_forecast_by_geographical_region_hourly(date=date_to_fetch)
        elif hasattr(iso, 'get_hourly_solar_report'):
            df_solar = iso.get_hourly_solar_report(date=date_to_fetch)
            
        if not df_solar.empty:
            # Columns: GEN CenterWest, STPPF CenterWest, PVGRPP CenterWest, COP HSL CenterWest
            # We want to extract all of them.
            
            time_col = 'Time' if 'Time' in df_solar.columns else df_solar.columns[1]
            
            # Identify regions from GEN columns
            gen_cols = [c for c in df_solar.columns if c.startswith('GEN ') and 'SYSTEM' not in c]
            regions = [c.replace('GEN ', '') for c in gen_cols]
            
            solar_records = []
            for _, row in df_solar.iterrows():
                ts = row[time_col]
                for region in regions:
                    solar_records.append({
                        'timestamp': ts,
                        'region': region,
                        'fuel_type': 'Solar',
                        'gen_mw': row.get(f'GEN {region}', np.nan),
                        'forecast_mw': row.get(f'STPPF {region}', np.nan),
                        'potential_mw': row.get(f'PVGRPP {region}', row.get(f'COP HSL {region}', np.nan))
                    })
            
            dfs.append(pd.DataFrame(solar_records))

    except Exception as e:
        print(f"    ⚠️  Error fetching Solar for {date_to_fetch}: {e}")

    # 2. Wind (Hourly Actuals + Forecasts)
    try:
        df_wind = pd.DataFrame()
        if hasattr(iso, 'get_wind_actual_and_forecast_by_geographical_region_hourly'):
            df_wind = iso.get_wind_actual_and_forecast_by_geographical_region_hourly(date=date_to_fetch)
        elif hasattr(iso, 'get_hourly_wind_report'):
            df_wind = iso.get_hourly_wind_report(date=date_to_fetch)

        if not df_wind.empty:
            time_col = 'Time' if 'Time' in df_wind.columns else df_wind.columns[1]
            
            # Columns: GEN LZ WEST, STWPF LZ WEST, WGRPP LZ WEST, COP HSL LZ WEST
            gen_cols = [c for c in df_wind.columns if c.startswith('GEN ') and 'SYSTEM' not in c]
            regions = [c.replace('GEN ', '') for c in gen_cols]
            
            wind_records = []
            for _, row in df_wind.iterrows():
                ts = row[time_col]
                for region in regions:
                    wind_records.append({
                        'timestamp': ts,
                        'region': region,
                        'fuel_type': 'Wind',
                        'gen_mw': row.get(f'GEN {region}', np.nan),
                        'forecast_mw': row.get(f'STWPF {region}', np.nan),
                        'potential_mw': row.get(f'WGRPP {region}', row.get(f'COP HSL {region}', np.nan))
                    })
            
            dfs.append(pd.DataFrame(wind_records))

    except Exception as e:
        print(f"    ⚠️  Error fetching Wind for {date_to_fetch}: {e}")
        
    if not dfs:
        return pd.DataFrame()
        
    return pd.concat(dfs, ignore_index=True)

def transform_and_map(df_gen: pd.DataFrame, assets: List[dict]) -> List[dict]:
    """Maps regional generation data to specific settlement points based on assets."""
    if df_gen.empty:
        return []
        
    # Pre-process assets for fast lookup
    # Asset -> (SolarSuffix, WindSuffix)
    asset_map = {}
    for asset in assets:
        sp = asset.get('settlement_point')
        zone = asset.get('inferred_zone')
        if sp:
            asset_map[sp] = get_zone_mapping(zone)
            
    records = []
    
    # Ensure timestamp is datetime and handle timezone to match ercot_prices (US/Central Naive)
    timestamps = pd.to_datetime(df_gen['timestamp'])
    if timestamps.dt.tz is not None:
        df_gen['timestamp'] = timestamps.dt.tz_convert('US/Central').dt.tz_localize(None)
    else:
        df_gen['timestamp'] = timestamps
    
    # Drop duplicates
    df_gen = df_gen.drop_duplicates(subset=['timestamp', 'region', 'fuel_type'])
    
    # Lookup dict: (timestamp, region, fuel_type) -> row_dict
    gen_lookup = df_gen.set_index(['timestamp', 'region', 'fuel_type']).to_dict('index')
    
    unique_timestamps = df_gen['timestamp'].unique()
    
    for sp, (solar_region, wind_region) in asset_map.items():
        for ts in unique_timestamps:
            # Add Solar Record
            solar_key = (ts, solar_region, 'Solar')
            if solar_key in gen_lookup:
                row = gen_lookup[solar_key]
                if pd.notna(row['gen_mw']):
                    records.append({
                        'timestamp': ts.isoformat(),
                        'settlement_point': sp,
                        'fuel_type': 'Solar',
                        'gen_mw': float(row['gen_mw']),
                        'forecast_mw': float(row['forecast_mw']) if pd.notna(row['forecast_mw']) else None,
                        'potential_mw': float(row['potential_mw']) if pd.notna(row['potential_mw']) else None,
                        'region': solar_region
                    })
                
            # Add Wind Record
            wind_key = (ts, wind_region, 'Wind')
            if wind_key in gen_lookup:
                row = gen_lookup[wind_key]
                if pd.notna(row['gen_mw']):
                    records.append({
                        'timestamp': ts.isoformat(),
                        'settlement_point': sp,
                        'fuel_type': 'Wind',
                        'gen_mw': float(row['gen_mw']),
                        'forecast_mw': float(row['forecast_mw']) if pd.notna(row['forecast_mw']) else None,
                        'potential_mw': float(row['potential_mw']) if pd.notna(row['potential_mw']) else None,
                        'region': wind_region
                    })
                
    return records


def upsert_to_supabase(supabase: Client, records: list[dict]) -> int:
    """Upserts records to ercot_generation table."""
    if not records:
        return 0
        
    total_inserted = 0
    try:
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i:i + BATCH_SIZE]
            supabase.table("ercot_generation").upsert(
                batch, on_conflict="timestamp,settlement_point,fuel_type"
            ).execute()
            total_inserted += len(batch)
    except Exception as e:
        print(f"    ✗ Batch insert failed: {str(e)[:150]}...")
        
    return total_inserted

def main():
    parser = argparse.ArgumentParser(description="Fetch ERCOT Generation Data")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=7, help="Number of past days to fetch if no dates provided")
    args = parser.parse_args()
    
    if not GRIDSTATUS_AVAILABLE:
        print("❌ Error: gridstatus library is required.")
        return 1
        
    load_dotenv()
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        print("❌ ERROR: Supabase credentials not found in .env file.")
        return 1
        
    supabase = create_client(supabase_url, supabase_key)
    
    # 1. Get Assets
    print("🔌 Fetching assets from Supabase...")
    try:
        res = supabase.table("engie_storage_assets").select("settlement_point, inferred_zone").execute()
        assets = res.data
        print(f"✅ Found {len(assets)} assets.")
    except Exception as e:
        print(f"❌ Failed to fetch assets: {e}")
        return 1
        
    if not assets:
        print("⚠️ No assets found. Exiting.")
        return 0
        
    print("\n📋 Processing the following Engie assets:")
    for a in assets:
        print(f"   - {a.get('settlement_point')} (Zone: {a.get('inferred_zone')})")
    print("-" * 50)

    # 2. Determine Date Range
    if args.start:
        start_date = datetime.datetime.strptime(args.start, "%Y-%m-%d").date()
    else:
        start_date = date.today() - timedelta(days=args.days)
        
    if args.end:
        end_date = datetime.datetime.strptime(args.end, "%Y-%m-%d").date()
    else:
        end_date = date.today()
        
    print(f"📅 Fetching generation data from {start_date} to {end_date}")
    
    iso = Ercot()
    print(f"DEBUG: gridstatus version: {getattr(gridstatus, '__version__', 'unknown')}")
    print(f"DEBUG: iso attributes: {[a for a in dir(iso) if 'solar' in a.lower() or 'wind' in a.lower()]}")
    
    # 3. Fetch Loop
    current_date = start_date
    total_records = 0
    
    # 3. Fetch Loop
    current_date = start_date
    total_records = 0
    
    print(f"DEBUG: Starting loop from {start_date} to {end_date}")
    
    while current_date <= end_date:
        print(f"\nProcessing {current_date}...")
        
        # Fetch Regional Data
        df_gen = fetch_generation_data(iso, current_date)
        print(f"  -> Fetched {len(df_gen)} rows of generation data")
        
        if not df_gen.empty:
            # Map to Assets
            records = transform_and_map(df_gen, assets)
            print(f"  -> Mapped to {len(records)} records for Supabase")
            
            # Upsert
            inserted = upsert_to_supabase(supabase, records)
            total_records += inserted
            print(f"  -> Inserted {inserted} records")
        else:
            print("  -> No data fetched")
            
        current_date += timedelta(days=1)
            
    print(f"\n✅ Completed. Total records inserted: {total_records}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
