"""
Migrate Existing CSV Data to Supabase
Zentus - ERCOT Battery Revenue Dashboard

Imports existing price data from CSV files into Supabase database.
Run this once after setting up the database schema.

Data sources:
- ERCOT-Dashboard/data/da_prices.csv
- ERCOT-Dashboard/data/rt_prices.csv
- Optionally: ../ERCOT-Live-Data/ercot_prices.csv (if available)

Usage:
    python scripts/migrate_existing_data.py
    python scripts/migrate_existing_data.py --include-live-data
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))


def load_dashboard_csv_data(data_dir: Path) -> pd.DataFrame:
    """
    Load and merge DA and RT CSV files from dashboard.

    Args:
        data_dir: Path to data directory

    Returns:
        Combined DataFrame with all price data
    """
    da_file = data_dir / "da_prices.csv"
    rt_file = data_dir / "rt_prices.csv"

    all_data = []

    # Load DA prices
    if da_file.exists():
        print(f"📂 Loading {da_file.name}...")
        da_df = pd.read_csv(da_file)
        da_df['market'] = 'DAM'
        da_df['interval_start'] = da_df['timestamp']
        da_df['interval_end'] = pd.to_datetime(da_df['timestamp']) + pd.Timedelta(hours=1)
        da_df['location'] = da_df['node']
        da_df['location_type'] = 'Resource Node'
        all_data.append(da_df)
        print(f"  ✓ Loaded {len(da_df):,} DAM records")
    else:
        print(f"⚠️  File not found: {da_file}")

    # Load RT prices
    if rt_file.exists():
        print(f"📂 Loading {rt_file.name}...")
        rt_df = pd.read_csv(rt_file)
        rt_df['market'] = 'RTM'
        rt_df['interval_start'] = rt_df['timestamp']
        rt_df['interval_end'] = pd.to_datetime(rt_df['timestamp']) + pd.Timedelta(hours=1)
        rt_df['location'] = rt_df['node']
        rt_df['location_type'] = 'Resource Node'
        all_data.append(rt_df)
        print(f"  ✓ Loaded {len(rt_df):,} RTM records")
    else:
        print(f"⚠️  File not found: {rt_file}")

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        print(f"✅ Total dashboard records: {len(combined):,}")
        return combined
    else:
        return pd.DataFrame()


def load_live_data_csv(live_data_dir: Path) -> pd.DataFrame:
    """
    Load CSV data from ERCOT-Live-Data directory if available.

    Args:
        live_data_dir: Path to ERCOT-Live-Data directory

    Returns:
        DataFrame with live data or empty DataFrame
    """
    ercot_file = live_data_dir / "ercot_prices.csv"

    if not ercot_file.exists():
        print(f"⚠️  Live data file not found: {ercot_file}")
        return pd.DataFrame()

    print(f"📂 Loading {ercot_file.name}...")
    df = pd.read_csv(ercot_file)

    # Transform to match database schema
    # Expected columns: Time, Interval Start, Interval End, Location, Location Type, Market, SPP
    df_transformed = df.rename(columns={
        'Time': 'timestamp',
        'Interval Start': 'interval_start',
        'Interval End': 'interval_end',
        'Location': 'location',
        'Location Type': 'location_type',
        'Market': 'market',
        'SPP': 'price_mwh'
    })

    print(f"  ✓ Loaded {len(df_transformed):,} live data records")
    return df_transformed


def transform_for_database(df: pd.DataFrame) -> list[dict]:
    """
    Transform DataFrame to match database schema.

    Args:
        df: Input DataFrame

    Returns:
        List of dictionaries for database insertion
    """
    if df.empty:
        return []

    # Ensure required columns exist
    required_cols = [
        'timestamp', 'interval_start', 'interval_end',
        'location', 'location_type', 'market', 'price_mwh'
    ]

    # Convert timestamp columns to datetime if not already
    for col in ['timestamp', 'interval_start', 'interval_end']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    # Select and order columns
    df_final = df[required_cols].copy()

    # Convert to list of dictionaries
    records = df_final.to_dict('records')

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
            # Upsert with conflict handling
            response = (
                supabase.table("ercot_prices")
                .upsert(batch, on_conflict="timestamp,interval_start,location,market")
                .execute()
            )

            total_inserted += len(batch)
            print(f"  ✓ Batch {batch_num}/{total_batches}: {len(batch)} records")

        except Exception as e:
            print(f"  ✗ Batch {batch_num}/{total_batches} failed: {e}")
            # Continue with next batch even if one fails
            continue

    print(f"✅ Successfully upserted {total_inserted:,} records")
    return total_inserted


def refresh_materialized_view(supabase: Client):
    """Refresh the materialized view for merged DAM/RTM prices."""
    print("🔄 Refreshing materialized view...")
    try:
        supabase.rpc('refresh_prices_merged').execute()
        print("✅ Materialized view refreshed")
    except Exception as e:
        print(f"⚠️  Could not refresh view: {e}")
        print("    Refresh manually in Supabase SQL Editor:")
        print("    REFRESH MATERIALIZED VIEW CONCURRENTLY ercot_prices_merged;")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Migrate existing CSV data to Supabase"
    )
    parser.add_argument(
        "--include-live-data",
        action="store_true",
        help="Also import data from ../ERCOT-Live-Data/ if available"
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    print("="*80)
    print("ERCOT Data Migration - Zentus")
    print("="*80)
    print()

    # Check Supabase credentials
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        print("❌ ERROR: Supabase credentials not found!")
        print("Set SUPABASE_URL and SUPABASE_KEY in .env file")
        return 1

    # Initialize Supabase client
    print("🔌 Connecting to Supabase...")
    try:
        supabase: Client = create_client(supabase_url, supabase_key)
        print("✅ Connected successfully")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return 1

    print()

    # Determine data directory paths
    script_dir = Path(__file__).parent
    dashboard_dir = script_dir.parent
    data_dir = dashboard_dir / "data"
    live_data_dir = dashboard_dir.parent / "ERCOT-Live-Data"

    all_data = []

    # Load dashboard CSV data
    print("Step 1: Loading dashboard CSV files...")
    print("-" * 80)
    dashboard_data = load_dashboard_csv_data(data_dir)
    if not dashboard_data.empty:
        all_data.append(dashboard_data)
    print()

    # Load live data if requested
    if args.include_live_data:
        print("Step 2: Loading live data CSV...")
        print("-" * 80)
        live_data = load_live_data_csv(live_data_dir)
        if not live_data.empty:
            all_data.append(live_data)
        print()
    else:
        print("Step 2: Skipping live data (use --include-live-data to import)")
        print()

    # Combine all data
    if not all_data:
        print("❌ No data found to migrate!")
        return 1

    print("Step 3: Combining and transforming data...")
    print("-" * 80)
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"✅ Total records to migrate: {len(combined_df):,}")

    # Remove duplicates
    initial_count = len(combined_df)
    combined_df = combined_df.drop_duplicates(
        subset=['timestamp', 'interval_start', 'location', 'market']
    )
    duplicates_removed = initial_count - len(combined_df)
    if duplicates_removed > 0:
        print(f"  ℹ️  Removed {duplicates_removed:,} duplicate records")

    # Transform for database
    records = transform_for_database(combined_df)
    print(f"✅ Prepared {len(records):,} unique records")
    print()

    # Insert into Supabase
    print("Step 4: Uploading to Supabase...")
    print("-" * 80)
    inserted = upsert_to_supabase(supabase, records)
    print()

    # Refresh materialized view
    print("Step 5: Refreshing materialized view...")
    print("-" * 80)
    refresh_materialized_view(supabase)
    print()

    print("="*80)
    print(f"✅ Migration complete! Migrated {inserted:,} records to Supabase")
    print("="*80)
    print()
    print("Next steps:")
    print("1. Verify data in Supabase dashboard")
    print("2. Run: python scripts/test_database_connection.py")
    print("3. Configure dashboard to use database (modify config/settings.py)")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
