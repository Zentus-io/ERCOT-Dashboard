
import os
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")

if not url or not key:
    print("Error: Supabase credentials not found.")
    exit()

client = create_client(url, key)

node = "ALVIN_RN"
start_date = "2025-11-19"
end_date = "2025-11-27"

print(f"Checking data for {node} from {start_date} to {end_date}...")

try:
    # Check count first
    query = client.table("ercot_generation").select("*", count="exact") \
        .eq("settlement_point", node) \
        .gte("timestamp", start_date) \
        .lte("timestamp", end_date)
        
    response = query.execute()
    print(f"Total rows found: {response.count}")
    
    if response.data:
        df = pd.DataFrame(response.data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print("\nData Summary (Rows per Day):")
        daily_counts = df.groupby(df['timestamp'].dt.date).size()
        for date, count in daily_counts.items():
            print(f"{date}: {count} rows")
    else:
        print("No data returned.")

except Exception as e:
    print(f"Error: {e}")
