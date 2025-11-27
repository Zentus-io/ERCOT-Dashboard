"""
Test Supabase Database Connection (Optimized V2)
Zentus - ERCOT Battery Revenue Dashboard

Verifies that:
1. Supabase credentials are configured correctly.
2. Database connection works.
3. Required tables with the new optimized schema exist.
4. Data can be queried successfully.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, List

from dotenv import load_dotenv
from postgrest.exceptions import APIError
from supabase import Client, create_client

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))


def test_connection(supabase: Client) -> bool:
    """Test basic connection to Supabase."""
    print("1. Testing basic connection...")
    try:
        # A simple query to check the connection and read permissions
        supabase.table("ercot_prices").select("settlement_point").limit(1).execute()
        print("   ✅ Connection successful")
        return True
    except APIError as e:
        print("   ❌ Connection failed. Check your SUPABASE_URL and SUPABASE_KEY.")
        print(f"      Error: {str(e)[:100]}...")
        return False


def test_tables_exist(supabase: Client) -> bool:
    """Verify required tables exist in the new schema."""
    print("2. Checking for required tables...")
    required_tables = ["ercot_prices", "eia_batteries"]
    all_exist = True
    for table in required_tables:
        try:
            supabase.table(table).select("*", count="exact").limit(0).execute()
            print(f"   ✅ Table '{table}' exists.")
        except APIError as e:
            print(f"   ❌ Table '{table}' not found. Did you run the schema setup SQL?")
            print(f"      Error: {str(e)[:100]}...")
            all_exist = False
    return all_exist


def test_data_exists(supabase: Client) -> bool:
    """Check if data has been loaded into the main table."""
    print("3. Checking for data in 'ercot_prices' table...")
    try:
        response = supabase.table("ercot_prices").select(
            "settlement_point", count="exact").execute()
        count = response.count or 0
        if count > 0:
            print(f"   ✅ Found {count:,} records in ercot_prices.")
            return True
        print("   ⚠️  No data found in 'ercot_prices' table.")
        print("      Run 'scripts/migrate_existing_data.py' or 'scripts/fetch_ercot_data.py'.")
        return False
    except APIError as e:
        print(f"   ❌ Query failed: {str(e)[:100]}...")
        return False


def test_query_performance(supabase: Client) -> bool:
    """Test a sample query similar to dashboard usage."""
    print("4. Testing a sample filtered query...")
    try:
        location_response = supabase.table("ercot_prices").select(
            "settlement_point").limit(1).execute()
        if not location_response.data:
            print("   ⚠️  No data available to perform a query test.")
            return True

        sample_location = str(location_response.data[0].get('settlement_point', ''))
        if not sample_location:
            print("   ⚠️  Could not retrieve a sample settlement_point.")
            return False

        start_time = datetime.now()
        response = (
            supabase.table("ercot_prices")
            .select("timestamp, price_mwh")
            .eq("settlement_point", sample_location)
            .eq("market", "DAM")
            .limit(100)
            .execute()
        )
        end_time = datetime.now()
        query_time = (end_time - start_time).total_seconds()

        print(f"   ✅ Query executed in {query_time:.3f} seconds.")
        print(
            f"   ✅ Retrieved {len(response.data)} records for settlement_point '{sample_location}'.")
        return True
    except APIError as e:
        print(f"   ❌ Query test failed: {str(e)[:100]}...")
        return False


def main():
    """Main test execution."""
    load_dotenv()
    print("=" * 80)
    print("Zentus - Supabase Database Connection Test (Optimized Schema V2)")
    print("=" * 80)

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        print("\n❌ ERROR: Supabase credentials not found!")
        print("   Please set SUPABASE_URL and SUPABASE_KEY in your .env file.")
        return 1

    print(f"\nSupabase URL: {supabase_url}")

    try:
        print("\n🔌 Initializing Supabase client...")
        supabase: Client = create_client(supabase_url, supabase_key)
        print("✅ Client initialized.")
    except Exception as e:  # pylint: disable=broad-except
        print(f"❌ Client initialization failed: {e}")
        return 1

    print("\n🚀 Running tests...")
    print("-" * 80)

    tests: List[Any] = [
        test_connection,
        test_tables_exist,
        test_data_exists,
        test_query_performance,
    ]

    results = []
    for test_func in tests:
        try:
            result = test_func(supabase)
            results.append(result)
            print()
        except Exception as e:  # pylint: disable=broad-except
            print(f"   ❌ Test error: {e}")
            print()
            results.append(False)

    # Summary
    print("=" * 80)
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"✅ All tests passed ({passed}/{total})")
        print("\nDatabase is ready for use with the new optimized schema!")
    else:
        print(f"⚠️  Some tests failed ({passed}/{total} passed)")
        print("\nPlease review the errors above and check your setup.")

    print("=" * 80)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
