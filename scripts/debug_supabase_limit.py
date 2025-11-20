
import sys
import os
from pathlib import Path
from datetime import date, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.data.loaders import SupabaseDataLoader
from config.settings import SUPABASE_URL, SUPABASE_KEY

def debug_supabase_loading():
    print(f"Connecting to Supabase: {SUPABASE_URL}")
    
    try:
        loader = SupabaseDataLoader()
        
        # Simulate the call made by the app
        print("\n--- Test 1: Loading ALL nodes (as done in app) ---")
        end_date = date.today()
        start_date = end_date - timedelta(days=5) # Just 5 days to be safe, but app uses 30
        
        print(f"Requesting data from {start_date} to {end_date}")
        
        # We need to access the raw query to see what's happening, 
        # but let's first call the method and see the result size
        try:
            df = loader.load_prices(node=None, start_date=start_date, end_date=end_date)
            print(f"Rows returned: {len(df)}")
            if not df.empty:
                print(f"Unique nodes found: {df['node'].nunique()}")
                print(f"Nodes: {df['node'].unique()[:10]}")
                print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            else:
                print("No data returned.")
                
        except Exception as e:
            print(f"Error calling load_prices: {e}")

        # Test 2: Check raw row count from a direct query to see if we hit the 1000 limit
        print("\n--- Test 2: Direct Query Limit Check ---")
        query = loader.client.table("ercot_prices").select("timestamp", count="exact").limit(10000)
        # query = query.gte("timestamp", start_date.isoformat())
        res = query.execute()
        print(f"Total rows in table (approx): {len(res.data)}")
        
        # Test 3: Fetch available nodes
        print("\n--- Test 3: Fetching Available Nodes ---")
        nodes = loader.get_available_nodes()
        print(f"Total available nodes: {len(nodes)}")
        print(f"First 5 nodes: {nodes[:5]}")

    except Exception as e:
        print(f"Setup failed: {e}")

if __name__ == "__main__":
    debug_supabase_loading()
