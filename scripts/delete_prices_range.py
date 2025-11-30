"""
Delete ERCOT Price Data Range
Zentus - ERCOT Battery Revenue Dashboard

Purpose:
    Deletes price data (DAM/RTM) from the `ercot_prices` table for a specific date range.
    Useful for cleaning up partial or incorrect data before re-fetching.

Usage:
    python scripts/delete_prices_range.py --start 2025-11-20 --end 2025-11-26
"""

import argparse
import os
import sys
from dotenv import load_dotenv
from supabase import create_client, Client

def main():
    parser = argparse.ArgumentParser(description="Delete ERCOT prices for a specific date range.")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    load_dotenv()
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        print("❌ Error: SUPABASE_URL or SUPABASE_KEY not found in .env")
        return

    print("🔌 Connecting to Supabase...")
    supabase: Client = create_client(url, key)

    print(f"⚠️  WARNING: This will delete ALL price data from {args.start} to {args.end}.")
    confirm = input("Type 'DELETE' to confirm: ")

    if confirm == "DELETE":
        print("🗑️  Deleting records...")
        try:
            # Delete where timestamp >= start and timestamp <= end (inclusive-ish)
            # Supabase timestamps are ISO strings in DB usually.
            # We'll use gte and lte.
            # Note: timestamp in DB includes time.
            # start date 00:00:00 to end date 23:59:59
            start_ts = f"{args.start}T00:00:00"
            end_ts = f"{args.end}T23:59:59"
            
            supabase.table("ercot_prices").delete() \
                .gte("timestamp", start_ts) \
                .lte("timestamp", end_ts) \
                .execute()
            print("✅ Data deleted successfully.")
        except Exception as e:
            print(f"❌ Error deleting data: {e}")
    else:
        print("❌ Operation cancelled.")

if __name__ == "__main__":
    main()
