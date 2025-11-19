"""
Optimized ERCOT Data Fetcher for Supabase (V3 - Smart Fetch)
Zentus - ERCOT Battery Revenue Dashboard

This script fetches Day-Ahead (DAM) and Real-Time (RTM) settlement prices
from the ERCOT API. It first queries the database to find incomplete days
within the specified range and only fetches data for those days, saving
significant time and API calls on re-runs.
"""

import os
import sys
import argparse
import datetime
from datetime import date, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client
from postgrest.types import CountMethod
from gridstatus.ercot import Ercot
from gridstatus.base import Markets, NoDataFoundException
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# ============================================================================ 
# CONFIGURATION
# ============================================================================ 

BATCH_SIZE = 1500
MAX_RETRIES = 3
PARALLEL_MARKETS = True
EXPECTED_RTM_RECORDS_PER_DAY = 96
EXPECTED_DAM_RECORDS_PER_DAY = 24


# ============================================================================ 
# HELPER FUNCTIONS
# ============================================================================ 

def fetch_ercot_prices_for_day(iso: Ercot, day: date, market: Markets, market_name: str) -> pd.DataFrame:
    """Fetches ERCOT prices for a single day and market."""
    try:
        prices = iso.get_spp(date=day, market=market)
        if not prices.empty:
            prices["Market"] = market_name
            return prices
    except NoDataFoundException:
        pass
    except Exception as e:
        print(f"    ⚠️  Error fetching {market_name} for {day}: {str(e)[:100]}...")
    return pd.DataFrame()


def transform_for_database(df: pd.DataFrame) -> list[dict]:
    """Transforms a DataFrame from gridstatus to match the V2 optimized database schema."""
    if df.empty:
        return []

    # Truncate timestamp to the minute
    df['timestamp'] = pd.to_datetime(df['Time']).dt.floor('min')

    df_transformed = df.rename(columns={
        'Location': 'settlement_point',
        'Market': 'market',
        'SPP': 'price_mwh'
    })

    required_cols = ['timestamp', 'settlement_point', 'market', 'price_mwh']
    
    # Ensure all required columns exist, add if not
    for col in required_cols:
        if col not in df_transformed.columns:
            df_transformed[col] = None

    records = df_transformed[required_cols].to_dict('records')

    # Final type casting for JSON compatibility
    for record in records:
        if isinstance(record['timestamp'], (datetime.datetime, pd.Timestamp)):
            record['timestamp'] = record['timestamp'].isoformat()
    return records


def upsert_to_supabase(supabase: Client, records: list[dict]) -> int:
    """Upserts a list of records to the ercot_prices table in batches."""
    if not records:
        return 0
    try:
        supabase.table("ercot_prices").upsert(
            records, on_conflict="timestamp,settlement_point,market"
        ).execute()
        return len(records)
    except Exception as e:
        print(f"    ✗ Batch insert failed: {str(e)[:100]}...")
        return 0


def fetch_and_store_day(iso: Ercot, supabase: Client, day: date, expected_counts: dict, retry_count: int = 0) -> dict:
    """Orchestrates fetching, transforming, and storing data for a single day."""
    stats = {'dam_records': 0, 'rtm_records': 0, 'total_inserted': 0, 'success': False}
    try:
        # If expected_counts is empty, it means we should fetch both markets for the day
        dam_needed = not expected_counts or expected_counts.get("DAM", 0) > 0
        rtm_needed = not expected_counts or expected_counts.get("RTM", 0) > 0

        dam_prices, rtm_prices = pd.DataFrame(), pd.DataFrame()

        if PARALLEL_MARKETS and (dam_needed and rtm_needed):
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_dam = executor.submit(fetch_ercot_prices_for_day, iso, day, Markets.DAY_AHEAD_HOURLY, "DAM")
                future_rtm = executor.submit(fetch_ercot_prices_for_day, iso, day, Markets.REAL_TIME_15_MIN, "RTM")
                dam_prices, rtm_prices = future_dam.result(), future_rtm.result()
        else:
            if dam_needed:
                dam_prices = fetch_ercot_prices_for_day(iso, day, Markets.DAY_AHEAD_HOURLY, "DAM")
            if rtm_needed:
                rtm_prices = fetch_ercot_prices_for_day(iso, day, Markets.REAL_TIME_15_MIN, "RTM")

        all_prices = [df for df in [dam_prices, rtm_prices] if not df.empty]
        stats['dam_records'], stats['rtm_records'] = len(dam_prices), len(rtm_prices)
        
        if all_prices:
            combined = pd.concat(all_prices, ignore_index=True)
            records = transform_for_database(combined)
            stats['total_inserted'] = upsert_to_supabase(supabase, records)
        stats['success'] = True
        return stats
    except Exception as e:
        if retry_count < MAX_RETRIES:
            return fetch_and_store_day(iso, supabase, day, expected_counts, retry_count + 1)
        else:
            print(f"    ❌ Day {day} failed after {MAX_RETRIES} retries: {str(e)[:100]}...")
            return stats


def find_incomplete_days(supabase: Client, start_date: date, end_date: date) -> List[date]:
    """
    Analyzes the database to find which days in the range are missing data.
    """
    print("\n🧐 Analyzing existing data in Supabase to find gaps...")
    # 1. Get the list of unique settlement points to calculate expected counts
    try:
        nodes_res = supabase.table("ercot_prices").select("settlement_point", count=CountMethod.exact).execute()
        unique_nodes = set(row['settlement_point'] for row in nodes_res.data if isinstance(row, dict) and row.get('settlement_point'))
        unique_nodes = set(row['settlement_point'] for row in nodes_res.data if isinstance(row, dict) and row.get('settlement_point'))
        num_nodes = len(unique_nodes)
        
        if num_nodes == 0:
            print("   -> No existing settlement points found. Will fetch all days in range.")
            return [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]

        print(f"   -> Found {num_nodes} unique settlement points in the database.")
    except Exception as e:
        print(f"   -> Could not get settlement points. Will fetch all days. Error: {e}")
        return [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]

    # 2. Get daily record counts from Supabase
    try:
        daily_counts_res = supabase.rpc('get_daily_summary', {
            'p_start_date': str(start_date),
            'p_end_date': str(end_date)
        }).execute()
        daily_counts_data = daily_counts_res.data
    except Exception as e:
         print(f"   -> Could not get daily summary. Will fetch all days. Error: {e}")
         return [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]

    if not isinstance(daily_counts_data, list):
        print(f"   -> Invalid daily summary data received. Will fetch all days.")
        return [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]

    # 3. Identify incomplete days
    all_days = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
    days_to_fetch = []
    
    counts_by_day = {}
    for item in daily_counts_data:
        if isinstance(item, dict) and 'day' in item and 'market' in item and 'record_count' in item:
            day_str = item['day']
            if isinstance(day_str, str):
                day_date = datetime.datetime.fromisoformat(day_str).date()
                market = item['market']
                if day_date not in counts_by_day:
                    counts_by_day[day_date] = {}
                counts_by_day[day_date][market] = item['record_count']
    
    expected_dam = num_nodes * EXPECTED_DAM_RECORDS_PER_DAY
    expected_rtm = num_nodes * EXPECTED_RTM_RECORDS_PER_DAY

    for day in all_days:
        day_counts = counts_by_day.get(day, {})
        dam_count = day_counts.get('DAM', 0)
        rtm_count = day_counts.get('RTM', 0)
        
        if dam_count < expected_dam or rtm_count < expected_rtm:
            days_to_fetch.append(day)

    if not days_to_fetch:
        print("   -> ✨ All days in the specified range are complete. No fetching needed.")
    else:
        print(f"   -> Found {len(days_to_fetch)} incomplete day(s) to fetch.")

    return days_to_fetch


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Optimized ERCOT data fetcher for Supabase.")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD). Default: 30 days ago.")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD). Default: today.")
    args = parser.parse_args()

    load_dotenv()
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        print("❌ ERROR: Supabase credentials not found in .env file.")
        return 1

    print("="*80)
    print("Zentus - Smart ERCOT Data Fetcher")
    print("="*80)

    try:
        print("\n🔌 Connecting to Supabase...")
        supabase: Client = create_client(supabase_url, supabase_key)
        print("✅ Supabase connected.")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return 1
        
    end_date = datetime.datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else date.today()
    start_date = datetime.datetime.strptime(args.start, "%Y-%m-%d").date() if args.start else end_date - timedelta(days=29)

    days_to_fetch = find_incomplete_days(supabase, start_date, end_date)

    if not days_to_fetch:
        return 0

    try:
        print("\n🔌 Initializing ERCOT API client...")
        ercot = Ercot()
        print("✅ ERCOT client initialized.")
    except Exception as e:
        print(f"❌ ERCOT client initialization failed: {e}")
        return 1

    print(f"\n📅 Fetching data for {len(days_to_fetch)} incomplete day(s)...")
    
    total_stats = {'dam_records': 0, 'rtm_records': 0, 'total_inserted': 0}
    failed_days_count = 0

    with tqdm(total=len(days_to_fetch), desc="Overall Progress", unit="day") as pbar:
        for day in days_to_fetch:
            pbar.set_description(f"Processing {day}")
            # The find_incomplete_days function already determined what's needed.
            # Pass an empty dict to signal fetch_and_store_day to try both markets if possible.
            stats = fetch_and_store_day(ercot, supabase, day, {})
            if stats['success']:
                total_stats['dam_records'] += stats['dam_records']
                total_stats['rtm_records'] += stats['rtm_records']
                total_stats['total_inserted'] += stats['total_inserted']
                pbar.set_postfix({'Inserted': stats['total_inserted']})
            else:
                failed_days_count += 1
                pbar.set_postfix({'Status': 'Failed'})
            pbar.update(1)

    print("\n" + "="*80)
    print("📜 Summary")
    print("="*80)
    print(f"✅ Successfully processed {len(days_to_fetch) - failed_days_count}/{len(days_to_fetch)} days.")
    print(f"📈 DAM records fetched: {total_stats['dam_records']:,}")
    print(f"📈 RTM records fetched: {total_stats['rtm_records']:,}")
    print(f"💾 Total records upserted: {total_stats['total_inserted']:,}")
    if failed_days_count > 0:
        print(f"❌ Failed days: {failed_days_count}")

    print("\n✅ Smart data fetch complete!")
    print("="*80)

    return 0

if __name__ == "__main__":
    sys.exit(main())
