"""Optimized ERCOT Data Fetcher for Supabase (V4.0 - Smart Gap Detection)
Zentus - ERCOT Battery Revenue Dashboard

This script intelligently fetches only missing Day-Ahead (DAM) and Real-Time (RTM)
settlement prices from the ERCOT API. It queries the database to find incomplete days,
fetches only what's needed, and reports any remaining gaps.

Key Features:
- Auto-detects date range from database (earliest to yesterday)
- Only fetches incomplete days (missing DAM or RTM data)
- Never attempts to fetch today's data (avoids infinite loops)
- Reports remaining gaps after fetch completion
"""

import os
import sys
import argparse
import datetime
from datetime import date, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client
from gridstatus.ercot import Ercot
from gridstatus.base import Markets, NoDataFoundException
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# ============================================================================
# CONFIGURATION
# ============================================================================

BATCH_SIZE = 250  # Further reduced to avoid timeouts with large datasets
MAX_RETRIES = 3
PARALLEL_MARKETS = True
EXPECTED_RTM_RECORDS_PER_DAY = 96
EXPECTED_DAM_RECORDS_PER_DAY = 24
DEFAULT_HISTORIC_START = date(2024, 1, 1)  # Fallback if database is empty


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

    # Convert to datetime, handling DST ambiguity by keeping US/Central time (ERCOT's timezone)
    timestamps = pd.to_datetime(df['Time'])
    # If timezone-aware, convert to Central time then remove tz; if naive, keep as-is
    if timestamps.dt.tz is not None:
        # Convert to US/Central (ERCOT's timezone), which handles DST automatically
        df['timestamp'] = timestamps.dt.tz_convert('US/Central').dt.tz_localize(None).dt.floor('min')
    else:
        df['timestamp'] = timestamps.dt.floor('min')

    df_transformed = df.rename(columns={
        'Location': 'settlement_point',
        'Market': 'market',
        'SPP': 'price_mwh'
    })

    required_cols = ['timestamp', 'settlement_point', 'market', 'price_mwh']
    
    for col in required_cols:
        if col not in df_transformed.columns:
            df_transformed[col] = None

    records = df_transformed[required_cols].to_dict('records')

    for record in records:
        if isinstance(record['timestamp'], (datetime.datetime, pd.Timestamp)):
            record['timestamp'] = record['timestamp'].isoformat()
    return records


def upsert_to_supabase(supabase: Client, records: list[dict]) -> int:
    """Upserts a list of records to the ercot_prices table in batches."""
    if not records:
        return 0

    # Deduplicate records within this batch to avoid "cannot affect row a second time" error
    # Keep last occurrence of each (timestamp, settlement_point, market) tuple
    seen = {}
    for record in records:
        key = (record['timestamp'], record['settlement_point'], record['market'])
        seen[key] = record
    deduplicated = list(seen.values())

    if len(deduplicated) < len(records):
        print(f"    ℹ️  Removed {len(records) - len(deduplicated)} duplicate records from batch")

    total_inserted = 0
    try:
        # Process in smaller batches
        for i in range(0, len(deduplicated), BATCH_SIZE):
            batch = deduplicated[i:i + BATCH_SIZE]
            supabase.table("ercot_prices").upsert(
                batch, on_conflict="timestamp,settlement_point,market"
            ).execute()
            total_inserted += len(batch)
        return total_inserted
    except Exception as e:
        print(f"    ✗ Batch insert failed at record {total_inserted}: {str(e)[:150]}...")
        return total_inserted  # Return what was successfully inserted before failure


def fetch_and_store_day(iso: Ercot, supabase: Client, day: date, expected_counts: dict, retry_count: int = 0) -> dict:
    """Orchestrates fetching, transforming, and storing data for a single day."""
    stats = {'dam_records': 0, 'rtm_records': 0, 'total_inserted': 0, 'success': False}
    try:
        dam_needed = True # Always try to fetch, upsert will handle conflicts
        rtm_needed = True

        dam_prices, rtm_prices = pd.DataFrame(), pd.DataFrame()

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_dam = executor.submit(fetch_ercot_prices_for_day, iso, day, Markets.DAY_AHEAD_HOURLY, "DAM")
            future_rtm = executor.submit(fetch_ercot_prices_for_day, iso, day, Markets.REAL_TIME_15_MIN, "RTM")
            dam_prices, rtm_prices = future_dam.result(), future_rtm.result()

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


def get_database_date_range(supabase: Client) -> tuple[date, date] | None:
    """
    Queries the database to find the earliest and latest dates with data.
    Returns (earliest_date, latest_date) or None if database is empty.
    """
    try:
        result = supabase.rpc('get_date_range').execute()
        if result.data and isinstance(result.data, list) and len(result.data) > 0:
            data = result.data[0]
            # Type guard: ensure data is a dict with the expected keys
            if isinstance(data, dict):
                min_date_str = data.get('min_date')
                max_date_str = data.get('max_date')
                if isinstance(min_date_str, str) and isinstance(max_date_str, str):
                    min_date = datetime.datetime.fromisoformat(min_date_str).date()
                    max_date = datetime.datetime.fromisoformat(max_date_str).date()
                    return (min_date, max_date)
    except Exception as e:
        print(f"   -> Could not get database date range: {e}")
    return None


def find_incomplete_days(supabase: Client, start_date: date, end_date: date) -> List[date]:
    """
    Analyzes the database to find which days in the range are missing data.
    """
    print("\n🧐 Analyzing existing data in Supabase to find gaps...")

    try:
        # Get unique settlement points without the count parameter to avoid type error
        nodes_res = supabase.table("ercot_prices").select("settlement_point").execute()
        unique_nodes = set(row['settlement_point'] for row in nodes_res.data if isinstance(row, dict) and row.get('settlement_point'))
        num_nodes = len(unique_nodes)
        
        if num_nodes == 0:
            print("   -> No existing settlement points found. Will fetch all days in range.")
            return [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]

        print(f"   -> Found {num_nodes} unique settlement points in the database.")
    except Exception as e:
        print(f"   -> Could not get settlement points. Will fetch all days. Error: {e}")
        return [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]

    # Use lightweight get_distinct_days function (much faster than get_daily_summary)
    try:
        distinct_days_res = supabase.rpc('get_distinct_days', {
            'p_start_date': str(start_date),
            'p_end_date': str(end_date)
        }).execute()

        if not distinct_days_res.data or not isinstance(distinct_days_res.data, list):
            print(f"   -> Could not get distinct days. Will fetch all days.")
            return [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]

        # Convert to set of dates for fast lookup
        existing_days = set()
        for item in distinct_days_res.data:
            if isinstance(item, dict) and 'day' in item:
                day_str = item['day']
                # Parse the date string (format: YYYY-MM-DD)
                if isinstance(day_str, str):
                    existing_days.add(datetime.datetime.strptime(day_str, "%Y-%m-%d").date())

        print(f"   -> Found {len(existing_days)} days with data in the database.")

        # Find days that are missing entirely
        all_days = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
        days_to_fetch = [day for day in all_days if day not in existing_days]

    except Exception as e:
        print(f"   -> Could not get distinct days (function may not exist). Error: {e}")
        print(f"   -> Run add_fast_gap_detection.sql to add the optimized function.")
        print(f"   -> Falling back to fetching all days in range.")
        return [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]

    if not days_to_fetch:
        print("   -> ✨ All days in the specified range are complete. No fetching needed.")
    else:
        print(f"   -> Found {len(days_to_fetch)} incomplete day(s) to fetch.")

    return days_to_fetch


def check_remaining_gaps(supabase: Client, start_date: date, end_date: date) -> Dict[str, Any]:
    """
    After fetching, checks if there are still any incomplete days in the range.
    Returns a dictionary with gap analysis results.
    """
    print("\n🔍 Validating data completeness...")

    remaining_incomplete = find_incomplete_days(supabase, start_date, end_date)

    result = {
        'has_gaps': len(remaining_incomplete) > 0,
        'incomplete_days': remaining_incomplete,
        'total_incomplete': len(remaining_incomplete)
    }

    if result['has_gaps']:
        print(f"\n⚠️  WARNING: {result['total_incomplete']} day(s) still have incomplete data:")

        # Group consecutive days for cleaner output
        if remaining_incomplete:
            ranges = []
            start = remaining_incomplete[0]
            prev = remaining_incomplete[0]

            for day in remaining_incomplete[1:]:
                if (day - prev).days == 1:
                    prev = day
                else:
                    ranges.append((start, prev) if start != prev else (start,))
                    start = day
                    prev = day
            ranges.append((start, prev) if start != prev else (start,))

            for r in ranges:
                if len(r) == 2:
                    print(f"   • {r[0]} to {r[1]}")
                else:
                    print(f"   • {r[0]}")

        print("\nPossible reasons:")
        print("   • ERCOT API may not have published data for these dates yet")
        print("   • Network/API errors during fetch (check logs above)")
        print("   • Data quality issues at settlement point level")
    else:
        print("   ✅ All days in range are now complete!")

    return result


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Smart ERCOT data fetcher that only fetches missing data from Supabase.",
        epilog="Examples:\n"
               "  python fetch_ercot_data.py                          # Fetch all missing data from DB start to yesterday\n"
               "  python fetch_ercot_data.py --start 2024-01-01       # Fetch from specific date to yesterday\n"
               "  python fetch_ercot_data.py --start 2024-01-01 --end 2024-12-31  # Fetch specific range",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD). Default: earliest date in database (or 2024-01-01 if empty).")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD). Default: yesterday (never today to avoid infinite loops).")
    args = parser.parse_args()

    load_dotenv()
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        print("❌ ERROR: Supabase credentials not found in .env file.")
        return 1

    print("="*80)
    print("Zentus - Smart ERCOT Data Fetcher (V4.0)")
    print("="*80)

    try:
        print("\n🔌 Connecting to Supabase...")
        supabase: Client = create_client(supabase_url, supabase_key)
        print("✅ Supabase connected.")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return 1

    # Always cap end_date at yesterday to avoid fetching today's incomplete data
    yesterday = date.today() - timedelta(days=1)
    if args.end:
        end_date = datetime.datetime.strptime(args.end, "%Y-%m-%d").date()
        if end_date > yesterday:
            print(f"⚠️  End date capped at yesterday ({yesterday}) to avoid fetching incomplete data.")
            end_date = yesterday
    else:
        end_date = yesterday

    # Auto-detect start_date from database if not specified
    if args.start:
        start_date = datetime.datetime.strptime(args.start, "%Y-%m-%d").date()
    else:
        db_range = get_database_date_range(supabase)
        if db_range:
            start_date = db_range[0]
            print(f"📅 Auto-detected database date range: {db_range[0]} to {db_range[1]}")
            print(f"   Using start date: {start_date}")
        else:
            start_date = DEFAULT_HISTORIC_START
            print(f"📅 Database is empty. Using default start date: {start_date}")

    print(f"\n📅 Target date range: {start_date} to {end_date}")
    print(f"   ({(end_date - start_date).days + 1} days total)")

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

    print(f"\n📅 Fetching data for {len(days_to_fetch)} incomplete day(s) existing data...")
    
    total_stats = {'dam_records': 0, 'rtm_records': 0, 'total_inserted': 0}
    failed_days_count = 0

    with tqdm(total=len(days_to_fetch), desc="Overall Progress", unit="day") as pbar:
        for day in days_to_fetch:
            pbar.set_description(f"Processing {day}")
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
    print("📜 Fetch Summary")
    print("="*80)
    print(f"✅ Successfully processed {len(days_to_fetch) - failed_days_count}/{len(days_to_fetch)} days.")
    print(f"📈 DAM records fetched: {total_stats['dam_records']:,}")
    print(f"📈 RTM records fetched: {total_stats['rtm_records']:,}")
    print(f"💾 Total records upserted: {total_stats['total_inserted']:,}")
    if failed_days_count > 0:
        print(f"❌ Failed days: {failed_days_count}")

    # Check for remaining gaps after fetch completion
    gap_report = check_remaining_gaps(supabase, start_date, end_date)

    print("\n" + "="*80)
    if gap_report['has_gaps']:
        print("⚠️  Fetch complete with gaps remaining")
        print("="*80)
        print(f"Run the script again to retry fetching the {gap_report['total_incomplete']} incomplete day(s).")
        return 1
    else:
        print("✅ Fetch complete - All data validated!")
        print("="*80)
        return 0

if __name__ == "__main__":
    sys.exit(main())
