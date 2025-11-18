"""
Fetch ERCOT Market Data and Store in Supabase
Zentus - ERCOT Battery Revenue Dashboard

Fetches Day-Ahead (DAM) and Real-Time (RTM) settlement prices from ERCOT API
using the gridstatus library and stores them in Supabase database.

Usage:
    # Fetch last 7 days
    python scripts/fetch_ercot_data.py

    # Fetch specific date range
    python scripts/fetch_ercot_data.py --start 2025-01-01 --end 2025-01-31

    # Refresh materialized view after fetching
    python scripts/fetch_ercot_data.py --refresh-view
"""

import os
import sys
import argparse
import datetime
from pathlib import Path
from typing import Optional
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client
from gridstatus.ercot import Ercot
from gridstatus.base import Markets, NoDataFoundException

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))


def fetch_ercot_prices(
    iso: Ercot,
    start_date: datetime.date,
    end_date: datetime.date
) -> pd.DataFrame:
    """
    Fetch DAM and RTM settlement prices from ERCOT API.

    Args:
        iso: ERCOT ISO instance
        start_date: Start date for data fetch
        end_date: End date for data fetch

    Returns:
        DataFrame with combined DAM and RTM prices
    """
    all_prices = []

    # Fetch DAM prices (hourly)
    try:
        print(f"📊 Fetching DAM prices from {start_date} to {end_date}...")
        dam_prices = iso.get_spp(
            start=start_date,
            end=end_date,
            market=Markets.DAY_AHEAD_HOURLY
        )
        if not dam_prices.empty:
            dam_prices["Market"] = "DAM"
            all_prices.append(dam_prices)
            print(f"✅ Fetched {len(dam_prices):,} DAM records")
        else:
            print("⚠️  No DAM data found for date range")
    except NoDataFoundException:
        print("⚠️  No DAM data available for date range")
    except Exception as e:
        print(f"❌ Error fetching DAM prices: {e}")

    # Fetch RTM prices (15-minute intervals)
    try:
        print(f"📊 Fetching RTM prices from {start_date} to {end_date}...")
        rtm_prices = iso.get_spp(
            start=start_date,
            end=end_date,
            market=Markets.REAL_TIME_15_MIN
        )
        if not rtm_prices.empty:
            rtm_prices["Market"] = "RTM"
            all_prices.append(rtm_prices)
            print(f"✅ Fetched {len(rtm_prices):,} RTM records")
        else:
            print("⚠️  No RTM data found for date range")
    except NoDataFoundException:
        print("⚠️  No RTM data available for date range")
    except Exception as e:
        print(f"❌ Error fetching RTM prices: {e}")

    if all_prices:
        combined = pd.concat(all_prices, ignore_index=True)
        print(f"✅ Total records fetched: {len(combined):,}")
        return combined
    else:
        return pd.DataFrame()


def transform_for_database(df: pd.DataFrame) -> list[dict]:
    """
    Transform ERCOT API data to match database schema.

    Args:
        df: DataFrame from gridstatus API

    Returns:
        List of dictionaries ready for database insertion
    """
    if df.empty:
        return []

    # Rename columns to match database schema
    # gridstatus columns: Time, Interval Start, Interval End, Location, Location Type, Market, SPP
    df_transformed = df.rename(columns={
        'Time': 'timestamp',
        'Interval Start': 'interval_start',
        'Interval End': 'interval_end',
        'Location': 'location',
        'Location Type': 'location_type',
        'Market': 'market',
        'SPP': 'price_mwh'
    })

    # Convert to list of dictionaries
    records = df_transformed[[
        'timestamp',
        'interval_start',
        'interval_end',
        'location',
        'location_type',
        'market',
        'price_mwh'
    ]].to_dict('records')

    # Convert timestamps to ISO format strings
    for record in records:
        record['timestamp'] = str(record['timestamp'])
        record['interval_start'] = str(record['interval_start'])
        record['interval_end'] = str(record['interval_end'])

    return records


def upsert_to_supabase(
    supabase: Client,
    records: list[dict],
    batch_size: int = 1000
) -> int:
    """
    Insert records into Supabase using batch upserts.

    Args:
        supabase: Supabase client
        records: List of price records
        batch_size: Number of records per batch

    Returns:
        Number of records inserted/updated
    """
    if not records:
        return 0

    total_inserted = 0
    total_batches = (len(records) + batch_size - 1) // batch_size

    print(f"💾 Inserting {len(records):,} records in {total_batches} batches...")

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        batch_num = (i // batch_size) + 1

        try:
            # Upsert with conflict handling on unique constraint
            response = (
                supabase.table("ercot_prices")
                .upsert(batch, on_conflict="timestamp,interval_start,location,market")
                .execute()
            )

            total_inserted += len(batch)
            print(f"  ✓ Batch {batch_num}/{total_batches}: {len(batch)} records")

        except Exception as e:
            print(f"  ✗ Batch {batch_num}/{total_batches} failed: {e}")
            continue

    print(f"✅ Successfully upserted {total_inserted:,} records")
    return total_inserted


def refresh_materialized_view(supabase: Client):
    """Refresh the materialized view for merged DAM/RTM prices."""
    print("🔄 Refreshing materialized view...")
    try:
        # Call the refresh function
        supabase.rpc('refresh_prices_merged').execute()
        print("✅ Materialized view refreshed")
    except Exception as e:
        print(f"⚠️  Could not refresh view: {e}")
        print("    You may need to refresh manually in Supabase SQL Editor:")
        print("    REFRESH MATERIALIZED VIEW CONCURRENTLY ercot_prices_merged;")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Fetch ERCOT price data and store in Supabase"
    )
    parser.add_argument(
        "--start",
        type=str,
        help="Start date (YYYY-MM-DD). Default: 7 days ago"
    )
    parser.add_argument(
        "--end",
        type=str,
        help="End date (YYYY-MM-DD). Default: today"
    )
    parser.add_argument(
        "--refresh-view",
        action="store_true",
        help="Refresh materialized view after fetching"
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Determine date range
    if args.end:
        end_date = datetime.datetime.strptime(args.end, "%Y-%m-%d").date()
    else:
        end_date = datetime.date.today()

    if args.start:
        start_date = datetime.datetime.strptime(args.start, "%Y-%m-%d").date()
    else:
        start_date = end_date - datetime.timedelta(days=7)

    print("="*80)
    print("ERCOT Data Fetcher - Zentus")
    print("="*80)
    print(f"Date range: {start_date} to {end_date}")
    print()

    # Check Supabase credentials
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        print("❌ ERROR: Supabase credentials not found!")
        print("Set SUPABASE_URL and SUPABASE_KEY in .env file")
        return 1

    # Check ERCOT API credentials (optional - gridstatus may work without)
    ercot_username = os.getenv("ERCOT_API_USERNAME")
    ercot_password = os.getenv("ERCOT_API_PASSWORD")
    ercot_key = os.getenv("ERCOT_API_SUBSCRIPTION_KEY")

    if not all([ercot_username, ercot_password, ercot_key]):
        print("⚠️  WARNING: ERCOT API credentials not fully configured")
        print("    Data fetching may be limited or fail")
        print("    Register at: https://apiexplorer.ercot.com/")
        print()

    # Initialize clients
    print("🔌 Connecting to Supabase...")
    try:
        supabase: Client = create_client(supabase_url, supabase_key)
        print("✅ Supabase connected")
    except Exception as e:
        print(f"❌ Supabase connection failed: {e}")
        return 1

    print("🔌 Initializing ERCOT API client...")
    try:
        ercot = Ercot()
        print("✅ ERCOT client initialized")
    except Exception as e:
        print(f"❌ ERCOT client initialization failed: {e}")
        return 1

    print()

    # Fetch data from ERCOT
    prices_df = fetch_ercot_prices(ercot, start_date, end_date)

    if prices_df.empty:
        print("❌ No data retrieved from ERCOT API")
        return 1

    # Transform data
    print()
    print("🔄 Transforming data for database...")
    records = transform_for_database(prices_df)
    print(f"✅ Prepared {len(records):,} records")

    # Insert into Supabase
    print()
    inserted = upsert_to_supabase(supabase, records)

    # Refresh materialized view if requested
    if args.refresh_view:
        print()
        refresh_materialized_view(supabase)

    print()
    print("="*80)
    print(f"✅ Data fetch complete! Inserted/updated {inserted:,} records")
    print("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
