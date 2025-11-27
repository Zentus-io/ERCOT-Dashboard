"""
Upload Engie Assets to Supabase
Zentus - ERCOT Battery Revenue Dashboard

Reads the Engie asset mapping CSV and uploads it to the 'engie_storage_assets' table in Supabase.
"""

import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from postgrest.exceptions import APIError
from supabase import Client, create_client

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))


def main():
    """Upload ENGIE asset data to Supabase database."""
    load_dotenv()
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        print("❌ ERROR: Supabase credentials not found in .env file.")
        return 1

    print("🔌 Connecting to Supabase...")
    try:
        supabase: Client = create_client(supabase_url, supabase_key)
        print("✅ Supabase connected.")
    except Exception as e:  # pylint: disable=broad-except
        print(f"❌ Initialization failed: {e}")
        return 1

    # Path to the CSV file
    # Using the path provided in the user request: DATA/results/engie_asset_mapping_final.csv
    # Assuming the script is run from the project root or scripts dir, we need to locate DATA relative to it.
    # The user said @[DATA/results/engie_asset_mapping_final.csv] is at
    # /home/boujuan/Documents/ZENTUS/ERCOT/DATA/results/engie_asset_mapping_final.csv

    csv_path = Path(
        "/home/boujuan/Documents/ZENTUS/ERCOT/DATA/results/engie_asset_mapping_final.csv")

    if not csv_path.exists():
        print(f"❌ ERROR: CSV file not found at {csv_path}")
        return 1

    print(f"📖 Reading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Map CSV columns to Supabase table columns
    # Table columns: plant_name, settlement_point, nameplate_power_mw, nameplate_energy_mwh,
    # duration_hours, county, inferred_zone, status_type, operating_year, resource_name, asset_id, hsl, lsl

    # CSV columns (from view_file):
    # Plant Name, Settlement Point, Nameplate Capacity (MW), Nameplate Energy Capacity (MWh),
    # County, Inferred_Zone, Status_Type, Operating Year, resource_name, asset_id, hsl, lsl

    # Replace NaN with None for JSON compatibility
    df = df.where(pd.notnull(df), None)

    records = []
    for _, row in df.iterrows():
        # Calculate duration if possible
        mw_val = row.get('Nameplate Capacity (MW)')
        mw = pd.to_numeric(mw_val, errors='coerce') if mw_val is not None else None

        mwh_val = row.get('Nameplate Energy Capacity (MWh)')
        mwh = pd.to_numeric(mwh_val, errors='coerce') if mwh_val is not None else None

        # Handle NaN from coercion
        if pd.isna(mw):
            mw = None
        if pd.isna(mwh):
            mwh = None

        duration = None
        if mw is not None and mwh is not None and mw > 0:
            duration = mwh / mw

        # Process hsl and lsl
        hsl_val = row.get('hsl')
        hsl = pd.to_numeric(hsl_val, errors='coerce') if hsl_val is not None else None
        if pd.isna(hsl):
            hsl = None

        lsl_val = row.get('lsl')
        lsl = pd.to_numeric(lsl_val, errors='coerce') if lsl_val is not None else None
        if pd.isna(lsl):
            lsl = None

        record = {
            'plant_name': row.get('Plant Name'),
            'settlement_point': row.get('Settlement Point'),
            'nameplate_power_mw': mw,
            'nameplate_energy_mwh': mwh,
            'duration_hours': duration,
            'county': row.get('County'),
            'inferred_zone': row.get('Inferred_Zone'),
            'status_type': row.get('Status_Type'),
            'operating_year': (int(row['Operating Year'])
                               if 'Operating Year' in row and pd.notna(row['Operating Year'])
                               else None),
            'resource_name': row.get('resource_name'),
            'asset_id': row.get('asset_id'),
            'hsl': hsl,
            'lsl': lsl}

        # Clean up record dictionary to ensure no NaNs remain
        for k, v in record.items():
            if pd.isna(v):
                record[k] = None

        records.append(record)

    print(f"🚀 Uploading {len(records)} records to 'engie_storage_assets'...")

    try:
        # Upsert data (using settlement_point as key if unique, but here we might just insert or upsert on id if we had it)
        # Since we don't have a unique constraint on settlement_point in the schema (it's not PK),
        # and we want to refresh the list, we might want to truncate first or just insert.
        # But `upsert` requires a unique constraint to work as update.
        # Let's just delete all and insert for a clean slate, as this is a reference table.

        print("   Cleaning existing data...")
        supabase.table("engie_storage_assets").delete().neq("id", -1).execute()  # Delete all

        print("   Inserting new data...")
        supabase.table("engie_storage_assets").insert(records).execute()

        print("✅ Upload complete!")
        return 0

    except APIError as e:
        print(f"❌ Upload failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
