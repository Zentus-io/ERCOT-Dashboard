
import os
import pandas as pd
from datetime import date
from dotenv import load_dotenv
from core.data.loaders import SupabaseDataLoader

load_dotenv()

def debug_loader():
    print("Initializing SupabaseDataLoader...")
    loader = SupabaseDataLoader()
    
    node = "ALVIN_RN"
    start_date = date(2025, 11, 19)
    end_date = date(2025, 11, 27)
    
    print(f"Calling load_prices for {node}...")
    try:
        df_prices = loader.load_prices(
            node=node,
            start_date=start_date,
            end_date=end_date
        )
        print(f"Prices Shape: {df_prices.shape}")
        print(f"Prices Index: {df_prices.index.min()} to {df_prices.index.max()}")
        if not df_prices.empty:
            print(f"Prices TZ: {df_prices.index.dtype}")
    except Exception as e:
        print(f"Error loading prices: {e}")
        return

    print(f"Calling load_generation_data for {node}...")
    try:
        df_gen = loader.load_generation_data(
            node=node,
            fuel_type="Solar",
            start_date=start_date,
            end_date=end_date
        )
        
        print(f"Gen Shape: {df_gen.shape}")
        if not df_gen.empty:
            print(f"Gen Index: {df_gen.index.min()} to {df_gen.index.max()}")
            print(f"Gen TZ: {df_gen.index.dtype}")
            
            # Simulate Alignment with Fix
            target_index = df_prices.index
            print("\nAttempting Alignment (with TZ Fix)...")
            
            # Fix: Normalize both to naive
            if df_gen.index.tz is not None:
                df_gen.index = df_gen.index.tz_convert(None)
            
            # Note: df_prices index is integer in this script because load_prices returns it that way.
            # But in the app, it's converted to datetime.
            # We need to simulate the app's state for df_prices.
            
            # Simulate app's df_prices index
            if 'timestamp' in df_prices.columns:
                df_prices['timestamp'] = pd.to_datetime(df_prices['timestamp'])
                # Assume naive for app simulation (as per loaders.py logic)
                if df_prices['timestamp'].dt.tz is not None:
                     df_prices['timestamp'] = df_prices['timestamp'].dt.tz_convert(None)
                target_index = pd.Index(df_prices['timestamp'])
            
            col_name = 'gen_mw'
            aligned = df_gen[col_name].reindex(target_index).interpolate(method='time').fillna(0)
            
            print(f"Aligned Shape: {aligned.shape}")
            print(f"Aligned Non-Zero Count: {(aligned > 0).sum()}")
            print("Aligned Daily Sums:")
            print(aligned.groupby(aligned.index.date).sum())
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_loader()
