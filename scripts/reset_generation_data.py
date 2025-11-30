"""
Reset ERCOT Generation Data
Zentus - ERCOT Battery Revenue Dashboard

Purpose:
    Deletes ALL data from the `ercot_generation` table.
    Used when a full reset is needed (e.g., after fixing timezone logic).

Usage:
    python scripts/reset_generation_data.py
"""

import os
import sys
from dotenv import load_dotenv
from supabase import create_client, Client

def main():
    load_dotenv()
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        print("❌ Error: SUPABASE_URL or SUPABASE_KEY not found in .env")
        return

    print("🔌 Connecting to Supabase...")
    supabase: Client = create_client(url, key)

    print("⚠️  WARNING: This will delete ALL data from the 'ercot_generation' table.")
    confirm = input("Type 'DELETE' to confirm: ")

    if confirm == "DELETE":
        print("🗑️  Deleting all records...")
        try:
            # Delete all rows (using not-null filter on a required column usually works for 'delete all' if no truncate RPC)
            # Or just use delete().neq('fuel_type', 'PLACEHOLDER')
            # Supabase-py delete requires a filter.
            # 'timestamp' is not null.
            supabase.table("ercot_generation").delete().neq("fuel_type", "PLACEHOLDER").execute()
            print("✅ Table cleared successfully.")
        except Exception as e:
            print(f"❌ Error deleting data: {e}")
            print("Tip: You can also run 'TRUNCATE TABLE ercot_generation;' in the Supabase SQL Editor.")
    else:
        print("❌ Operation cancelled.")

if __name__ == "__main__":
    main()
