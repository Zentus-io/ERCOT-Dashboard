"""
Migrate Existing CSV Data to Supabase (Optimized V2)
Zentus - ERCOT Battery Revenue Dashboard

Imports existing price data from local CSV files into the optimized
Supabase database schema.
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client
import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))


def load_dashboard_csv_data(data_dir: Path) -> pd.DataFrame:
    """Loads and transforms CSV data from the dashboard's /data folder."""
    da_file = data_dir / "da_prices.csv"
    rt_file = data_dir / "rt_prices.csv"
    all_data = []

    if da_file.exists():
        print(f"📂 Loading {da_file.name}...")
        da_df = pd.read_csv(da_file)
        da_df['market'] = 'DAM'
        all_data.append(da_df)
        print(f"  ✓ Loaded {len(da_df):,} DAM records")

    if rt_file.exists():
        print(f"📂 Loading {rt_file.name}...")
        rt_df = pd.read_csv(rt_file)
        rt_df['market'] = 'RTM'
        all_data.append(rt_df)
        print(f"  ✓ Loaded {len(rt_df):,} RTM records")

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        return combined
    return pd.DataFrame()


def transform_for_database(df: pd.DataFrame) -> list[dict]:
    """Transforms a DataFrame to match the V2 optimized database schema."""
    if df.empty:
        return []
    
    # Truncate timestamp to the minute
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.floor('min')

    df_transformed = df.rename(columns={
        'node': 'settlement_point',
        'price_mwh': 'price_mwh'
    })
    
    required_cols = ['timestamp', 'settlement_point', 'market', 'price_mwh']
    
    # Ensure all required columns exist
    for col in required_cols:
        if col not in df_transformed.columns:
            df_transformed[col] = None
    
    records = df_transformed[required_cols].to_dict('records')

    # Final type casting for JSON compatibility
    for record in records:
        if isinstance(record['timestamp'], (datetime.datetime, pd.Timestamp)):
            record['timestamp'] = record['timestamp'].isoformat()
    return records


def upsert_to_supabase(supabase: Client, records: list[dict], batch_size: int = 1500) -> int:
    """Upserts records to Supabase in batches."""
    if not records:
        return 0

    total_inserted = 0
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        try:
            supabase.table("ercot_prices").upsert(
                batch, on_conflict="timestamp,settlement_point,market"
            ).execute()
            total_inserted += len(batch)
        except Exception as e:
            print(f"    ✗ Batch insert failed: {str(e)[:100]}...")
            continue
    return total_inserted


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Migrate existing CSV data to Supabase.")
    args = parser.parse_args()

    load_dotenv()
    print("="*80)
    print("Zentus - CSV to Supabase Data Migration (Optimized Schema)")
    print("="*80)

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        print("\n❌ ERROR: Supabase credentials not found in .env file.")
        return 1

    print("\n🔌 Connecting to Supabase...")
    try:
        supabase: Client = create_client(supabase_url, supabase_key)
        print("✅ Connected successfully.")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return 1

    # Define paths
    dashboard_dir = Path(__file__).parent.parent
    data_dir = dashboard_dir / "data"

    print("\nStep 1: Loading and combining dashboard CSV files...")
    combined_df = load_dashboard_csv_data(data_dir)
    
    if combined_df.empty:
        print("\n❌ No data found to migrate!")
        return 1

    # Remove duplicates
    initial_count = len(combined_df)
    combined_df.drop_duplicates(subset=['timestamp', 'node', 'market'], inplace=True)
    duplicates_removed = initial_count - len(combined_df)
    if duplicates_removed > 0:
        print(f"  ℹ️  Removed {duplicates_removed:,} duplicate records from source CSVs.")

    print("\nStep 2: Transforming data for database...")
    records = transform_for_database(combined_df)
    print(f"✅ Prepared {len(records):,} unique records for upload.")

    print("\nStep 3: Uploading to Supabase...")
    inserted = upsert_to_supabase(supabase, records)

    print("\n" + "="*80)
    print(f"✅ Migration complete! Migrated {inserted:,} records.")
    print("="*80)

    return 0

if __name__ == "__main__":
    sys.exit(main())
