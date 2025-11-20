
import sys
import os
from pathlib import Path
from datetime import date, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.data.loaders import SupabaseDataLoader
from config.settings import SUPABASE_URL, SUPABASE_KEY

def verify_fix():
    print(f"Connecting to Supabase: {SUPABASE_URL}")
    
    try:
        loader = SupabaseDataLoader()
        
        # Step 1: Fetch available nodes
        print("\n--- Step 1: Fetching Available Nodes ---")
        nodes = loader.get_available_nodes()
        print(f"Total available nodes: {len(nodes)}")
        
        if not nodes:
            print("ERROR: No nodes found.")
            return

        # Step 2: Pick a node and fetch prices
        test_node = nodes[0] # Pick the first one
        print(f"\n--- Step 2: Fetching prices for node '{test_node}' ---")
        
        end_date = date.today()
        start_date = end_date - timedelta(days=30) 
        
        print(f"Requesting data from {start_date} to {end_date}")
        
        df = loader.load_prices(node=test_node, start_date=start_date, end_date=end_date)
        
        print(f"Rows returned: {len(df)}")
        if not df.empty:
            print(f"Data columns: {df.columns.tolist()}")
            print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print("SUCCESS: Data loaded for specific node.")
        else:
            print("WARNING: No data returned for this node (might be expected if no recent data).")
            
            # Try another node if the first one is empty, just in case
            if len(nodes) > 1:
                test_node = nodes[1]
                print(f"\n--- Retrying with second node '{test_node}' ---")
                df = loader.load_prices(node=test_node, start_date=start_date, end_date=end_date)
                print(f"Rows returned: {len(df)}")
                if not df.empty:
                     print("SUCCESS: Data loaded for second node.")

    except Exception as e:
        print(f"Verification failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_fix()
