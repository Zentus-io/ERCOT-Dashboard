"""
Optimized ERCOT Data Fetcher for Supabase
Zentus - ERCOT Battery Revenue Dashboard

This script fetches Day-Ahead (DAM) and Real-Time (RTM) settlement prices
from the ERCOT API using the gridstatus library and upserts them into a
Supabase database.

It is optimized for fetching large date ranges by processing data in
parallel chunks with progress tracking and retry logic.

Usage:
    # Fetch last 7 days (default)
    python scripts/fetch_ercot_data.py

    # Fetch a specific date range
    python scripts/fetch_ercot_data.py --start 2025-01-01 --end 2025-12-31

    # Just refresh the materialized view
    python scripts/fetch_ercot_data.py --refresh-view-only
"""

import os
import sys
import argparse
import datetime
from datetime import date, timedelta
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
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

CHUNK_DAYS = 7       # Fetch data in weekly chunks to avoid API timeouts
BATCH_SIZE = 1500    # Number of records to upsert in a single Supabase batch
MAX_RETRIES = 3      # Number of retries for a failed chunk
PARALLEL_MARKETS = True # Fetch DAM and RTM markets in parallel for each chunk


# ============================================================================ 
# HELPER FUNCTIONS
# ============================================================================ 

def chunk_date_range(start_date: date, end_date: date, chunk_days: int):
    """Splits a date range into smaller chunks."""
    current = start_date
    while current <= end_date:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end_date)
        yield (current, chunk_end)
        current = chunk_end + timedelta(days=1)


def fetch_ercot_prices_chunk(
    iso: Ercot, start_date: date, end_date: date, market: Markets, market_name: str
) -> pd.DataFrame:
    """Fetches ERCOT prices for a specific date chunk and market."""
    try:
        prices = iso.get_spp(start=start_date, end=end_date, market=market)
        if not prices.empty:
            prices["Market"] = market_name
            return prices
    except NoDataFoundException:
        # This is expected for dates where ERCOT has not published data
        pass
    except Exception as e:
        # Log other, unexpected errors
        print(f"    ⚠️  Error fetching {market_name} for {start_date}-{end_date}: {str(e)[:100]}...")
    return pd.DataFrame()


def transform_for_database(df: pd.DataFrame) -> list[dict]:
    """Transforms a DataFrame from gridstatus to match the Supabase schema."""
    if df.empty:
        return []

    df_transformed = df.rename(columns={
        'Time': 'timestamp',
        'Interval Start': 'interval_start',
        'Interval End': 'interval_end',
        'Location': 'location',
        'Location Type': 'location_type',
        'Market': 'market',
        'SPP': 'price_mwh'
    })

    # Ensure all required columns are present
    required_cols = ['timestamp', 'interval_start', 'interval_end', 'location', 'location_type', 'market', 'price_mwh']
    for col in required_cols:
        if col not in df_transformed.columns:
            df_transformed[col] = pd.NaT if 'time' in col else None
            
    records = df_transformed[required_cols].to_dict('records')

    for record in records:
        for key, value in record.items():
            if pd.isna(value):
                record[key] = None
            elif isinstance(value, (datetime.datetime, pd.Timestamp)):
                record[key] = value.isoformat()
    return records


def upsert_to_supabase(supabase: Client, records: list[dict]) -> int:
    """Upserts a list of records to the ercot_prices table in batches."""
    if not records:
        return 0

    total_inserted = 0
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        try:
            supabase.table("ercot_prices").upsert(
                batch, on_conflict="timestamp,interval_start,location,market"
            ).execute()
            total_inserted += len(batch)
        except Exception as e:
            print(f"    ✗ Batch insert failed: {str(e)[:100]}...")
            continue
    return total_inserted


def fetch_and_store_chunk(
    iso: Ercot, supabase: Client, chunk_start: date, chunk_end: date, retry_count: int = 0
) -> dict:
    """Orchestrates fetching, transforming, and storing data for one chunk."""
    stats = {'dam_records': 0, 'rtm_records': 0, 'total_inserted': 0, 'success': False}
    try:
        if PARALLEL_MARKETS:
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_dam = executor.submit(fetch_ercot_prices_chunk, iso, chunk_start, chunk_end, Markets.DAY_AHEAD_HOURLY, "DAM")
                future_rtm = executor.submit(fetch_ercot_prices_chunk, iso, chunk_start, chunk_end, Markets.REAL_TIME_15_MIN, "RTM")
                dam_prices, rtm_prices = future_dam.result(), future_rtm.result()
        else:
            dam_prices = fetch_ercot_prices_chunk(iso, chunk_start, chunk_end, Markets.DAY_AHEAD_HOURLY, "DAM")
            rtm_prices = fetch_ercot_prices_chunk(iso, chunk_start, chunk_end, Markets.REAL_TIME_15_MIN, "RTM")

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
            return fetch_and_store_chunk(iso, supabase, chunk_start, chunk_end, retry_count + 1)
        else:
            print(f"    ❌ Chunk failed after {MAX_RETRIES} retries: {str(e)[:100]}...")
            return stats

def refresh_materialized_view(supabase: Client):
    """Refreshes the materialized view for merged DAM/RTM prices."""
    print("\n🔄 Refreshing materialized view 'ercot_prices_merged'...")
    try:
        supabase.rpc('refresh_prices_merged').execute()
        print("✅ Materialized view refreshed successfully.")
    except Exception as e:
        print(f"⚠️  Could not refresh view automatically: {str(e)[:100]}...")
        print("    Please run this SQL in the Supabase SQL Editor:")
        print("    REFRESH MATERIALIZED VIEW CONCURRENTLY ercot_prices_merged;")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Optimized ERCOT data fetcher for Supabase.")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD). Default: 7 days ago.")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD). Default: today.")
    parser.add_argument("--refresh-view-only", action="store_true", help="Only refresh the materialized view and exit.")
    args = parser.parse_args()

    load_dotenv()
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        print("❌ ERROR: Supabase credentials not found in .env file.")
        return 1

    print("="*80)
    print("Zentus - Optimized ERCOT Data Fetcher")
    print("="*80)

    try:
        print("\n🔌 Connecting to Supabase...")
        supabase: Client = create_client(supabase_url, supabase_key)
        print("✅ Supabase connected.")
        
        if args.refresh_view_only:
            refresh_materialized_view(supabase)
            return 0
        
        print("\n🔌 Initializing ERCOT API client...")
        ercot = Ercot()
        print("✅ ERCOT client initialized.")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return 1

    end_date = datetime.datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else date.today()
    start_date = datetime.datetime.strptime(args.start, "%Y-%m-%d").date() if args.start else end_date - timedelta(days=6)

    chunks = list(chunk_date_range(start_date, end_date, CHUNK_DAYS))
    print(f"\n📅 Fetching data from {start_date} to {end_date} in {len(chunks)} chunk(s).")
    
    total_stats = {'dam_records': 0, 'rtm_records': 0, 'total_inserted': 0}
    failed_chunks_count = 0

    with tqdm(total=len(chunks), desc="Overall Progress", unit="chunk") as pbar:
        for chunk_start, chunk_end in chunks:
            pbar.set_description(f"Processing {chunk_start} to {chunk_end}")
            stats = fetch_and_store_chunk(ercot, supabase, chunk_start, chunk_end)
            if stats['success']:
                total_stats['dam_records'] += stats['dam_records']
                total_stats['rtm_records'] += stats['rtm_records']
                total_stats['total_inserted'] += stats['total_inserted']
                pbar.set_postfix({'Inserted': stats['total_inserted']})
            else:
                failed_chunks_count += 1
                pbar.set_postfix({'Status': 'Failed'})
            pbar.update(1)

    if total_stats['total_inserted'] > 0:
        refresh_materialized_view(supabase)

    print("\n" + "="*80)
    print("📜 Summary")
    print("="*80)
    print(f"✅ Successfully processed {len(chunks) - failed_chunks_count}/{len(chunks)} chunks.")
    print(f"📈 DAM records fetched: {total_stats['dam_records']:,}")
    print(f"📈 RTM records fetched: {total_stats['rtm_records']:,}")
    print(f"💾 Total records inserted/updated: {total_stats['total_inserted']:,}")
    if failed_chunks_count > 0:
        print(f"❌ Failed chunks: {failed_chunks_count}")

    print("\n✅ Data fetch complete!")
    print("="*80)

    return 0

if __name__ == "__main__":
    sys.exit(main())