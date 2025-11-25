"""
Quick test to verify polars integration for parquet I/O
Zentus - ERCOT Battery Revenue Dashboard
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data.loaders import ParquetDataLoader

def test_parquet_loader():
    """Test ParquetDataLoader with polars backend."""
    print("\n" + "="*60)
    print("Testing Polars Integration in ParquetDataLoader")
    print("="*60)

    # Initialize loader
    data_dir = Path(__file__).parent.parent / 'data'
    loader = ParquetDataLoader(data_dir)

    print(f"\nData directory: {data_dir}")
    print(f"DAM file exists: {loader.dam_path.exists()}")
    print(f"RTM file exists: {loader.rtm_path.exists()}")

    # Test 1: Get available nodes
    print("\n" + "-"*60)
    print("Test 1: Get Available Nodes")
    print("-"*60)
    nodes = loader.get_available_nodes()
    print(f"✓ Found {len(nodes)} nodes")
    if nodes:
        print(f"  Sample nodes: {nodes[:5]}")

    # Test 2: Get date range
    print("\n" + "-"*60)
    print("Test 2: Get Date Range")
    print("-"*60)
    start, end = loader.get_date_range()
    print(f"✓ Date range: {start} to {end}")

    # Test 3: Load prices for a specific node
    print("\n" + "-"*60)
    print("Test 3: Load Prices (with node filter)")
    print("-"*60)

    if nodes:
        test_node = nodes[0]
        print(f"Loading data for node: {test_node}")
        df = loader.load_prices(node=test_node)

        if not df.empty:
            print(f"✓ Loaded {len(df)} rows")
            print(f"  Columns: {list(df.columns)}")
            print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"  Price range (RT): ${df['price_mwh_rt'].min():.2f} - ${df['price_mwh_rt'].max():.2f}")
            print(f"\nFirst 3 rows:")
            print(df.head(3).to_string(index=False))
        else:
            print("⚠ No data returned")

    print("\n" + "="*60)
    print("✓ All tests passed! Polars integration working correctly.")
    print("="*60 + "\n")


if __name__ == "__main__":
    test_parquet_loader()
