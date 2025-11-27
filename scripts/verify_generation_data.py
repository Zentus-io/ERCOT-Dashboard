import os
import sys
import pandas as pd
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

    start_date = "2025-11-13"
    end_date = "2025-11-26"

    print(f"🔍 Verifying data from {start_date} to {end_date}...")

    # Fetch all data for the range (might be large, but for 2 weeks it's okay)
    # We'll just fetch counts per day/fuel_type/settlement_point if possible, 
    # but Supabase JS client doesn't do group by easily.
    # We'll fetch all and aggregate in Pandas.
    
    # Fetch all data for the range using pagination
    all_data = []
    offset = 0
    limit = 1000
    
    while True:
        print(f"    Fetching batch from offset {offset}...")
        res = supabase.table("ercot_generation") \
            .select("timestamp, settlement_point, fuel_type") \
            .gte("timestamp", f"{start_date}T00:00:00") \
            .lte("timestamp", f"{end_date}T23:59:59") \
            .range(offset, offset + limit - 1) \
            .execute()
            
        batch = res.data
        if not batch:
            break
            
        all_data.extend(batch)
        offset += limit
        
        if len(batch) < limit:
            break
            
    if not all_data:
        print("❌ No data found in range.")
        return

    df = pd.DataFrame(all_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date

    print(f"✅ Fetched {len(df)} records.")

    # Check completeness
    # Expected: 24 hours * N assets * 2 fuel types (if applicable)
    # Actually, let's just see counts per day.
    
    daily_counts = df.groupby(['date', 'fuel_type']).size().unstack(fill_value=0)
    print("\n📊 Daily Record Counts (Rows per Day):")
    print(daily_counts)

    # Check settlement points
    unique_sps = df['settlement_point'].unique()
    print(f"\n📍 Found {len(unique_sps)} unique settlement points:")
    print(sorted(unique_sps))
    
    # Check for missing days
    expected_days = pd.date_range(start=start_date, end=end_date).date
    found_days = sorted(df['date'].unique())
    
    missing_days = [d for d in expected_days if d not in found_days]
    
    if missing_days:
        print(f"\n❌ Missing Days: {missing_days}")
    else:
        print("\n✅ All days in range are present.")


if __name__ == "__main__":
    main()
