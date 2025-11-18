"""
Test Supabase Database Connection
Zentus - ERCOT Battery Revenue Dashboard

Verifies that:
1. Supabase credentials are configured correctly
2. Database connection works
3. Required tables exist
4. Sample queries execute successfully

Usage:
    python scripts/test_database_connection.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))


def test_connection(supabase: Client) -> bool:
    """Test basic connection to Supabase."""
    print("1. Testing basic connection...")
    try:
        # Try a simple query to check connection
        response = supabase.table("ercot_prices").select("*").limit(1).execute()
        print("   ✅ Connection successful")
        return True
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return False


def test_tables_exist(supabase: Client) -> bool:
    """Verify required tables exist."""
    print("2. Checking for required tables...")

    required_tables = ["ercot_prices", "eia_batteries"]
    all_exist = True

    for table in required_tables:
        try:
            response = supabase.table(table).select("*").limit(1).execute()
            print(f"   ✅ Table '{table}' exists")
        except Exception as e:
            print(f"   ❌ Table '{table}' not found: {e}")
            all_exist = False

    return all_exist


def test_data_exists(supabase: Client) -> bool:
    """Check if data has been loaded."""
    print("3. Checking for data in ercot_prices table...")

    try:
        response = supabase.table("ercot_prices").select("*", count="exact").limit(1).execute()

        count = response.count if hasattr(response, 'count') else len(response.data)

        if count > 0:
            print(f"   ✅ Found {count:,} records in ercot_prices")
            return True
        else:
            print("   ⚠️  No data found in ercot_prices table")
            print("      Run: python scripts/migrate_existing_data.py")
            return False

    except Exception as e:
        print(f"   ❌ Query failed: {e}")
        return False


def test_query_performance(supabase: Client) -> bool:
    """Test a sample query similar to dashboard usage."""
    print("4. Testing sample query (location + time range filter)...")

    try:
        # Get a sample location first
        location_response = (
            supabase.table("ercot_prices")
            .select("location")
            .limit(1)
            .execute()
        )

        if not location_response.data:
            print("   ⚠️  No data to test query")
            return False

        sample_location = location_response.data[0]['location']

        # Test a filtered query
        start_time = datetime.now()

        response = (
            supabase.table("ercot_prices")
            .select("*")
            .eq("location", sample_location)
            .eq("market", "DAM")
            .limit(100)
            .execute()
        )

        end_time = datetime.now()
        query_time = (end_time - start_time).total_seconds()

        print(f"   ✅ Query executed in {query_time:.3f} seconds")
        print(f"   ✅ Retrieved {len(response.data)} records for location '{sample_location}'")

        return True

    except Exception as e:
        print(f"   ❌ Query test failed: {e}")
        return False


def test_available_locations(supabase: Client) -> bool:
    """List available settlement points."""
    print("5. Listing available locations...")

    try:
        # Get distinct locations (Supabase doesn't support DISTINCT easily, so we do it client-side)
        response = (
            supabase.table("ercot_prices")
            .select("location, market")
            .limit(1000)
            .execute()
        )

        if response.data:
            locations = set(row['location'] for row in response.data)
            markets = set(row['market'] for row in response.data)

            print(f"   ✅ Found {len(locations)} unique locations")
            print(f"   ✅ Markets: {', '.join(sorted(markets))}")

            # Show first few locations
            if locations:
                sample_locs = list(sorted(locations))[:5]
                print(f"   Sample locations: {', '.join(sample_locs)}")

            return True
        else:
            print("   ⚠️  No locations found")
            return False

    except Exception as e:
        print(f"   ❌ Failed to list locations: {e}")
        return False


def test_date_range(supabase: Client) -> bool:
    """Check available date range in database."""
    print("6. Checking available date range...")

    try:
        # Get min and max timestamps
        response = (
            supabase.table("ercot_prices")
            .select("timestamp")
            .order("timestamp", desc=False)
            .limit(1)
            .execute()
        )

        if not response.data:
            print("   ⚠️  No timestamp data")
            return False

        earliest = response.data[0]['timestamp']

        response = (
            supabase.table("ercot_prices")
            .select("timestamp")
            .order("timestamp", desc=True)
            .limit(1)
            .execute()
        )

        latest = response.data[0]['timestamp']

        print(f"   ✅ Earliest data: {earliest}")
        print(f"   ✅ Latest data: {latest}")

        return True

    except Exception as e:
        print(f"   ❌ Date range check failed: {e}")
        return False


def main():
    """Main test execution."""
    # Load environment variables
    load_dotenv()

    print("="*80)
    print("Supabase Connection Test - Zentus")
    print("="*80)
    print()

    # Check credentials
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        print("❌ ERROR: Supabase credentials not found!")
        print("Set SUPABASE_URL and SUPABASE_KEY in .env file")
        print("See .env.example for template")
        return 1

    print(f"Supabase URL: {supabase_url}")
    print()

    # Initialize client
    print("Initializing Supabase client...")
    try:
        supabase: Client = create_client(supabase_url, supabase_key)
        print("✅ Client initialized")
    except Exception as e:
        print(f"❌ Client initialization failed: {e}")
        return 1

    print()
    print("Running tests...")
    print("-" * 80)

    # Run all tests
    tests = [
        test_connection,
        test_tables_exist,
        test_data_exists,
        test_query_performance,
        test_available_locations,
        test_date_range,
    ]

    results = []
    for test_func in tests:
        try:
            result = test_func(supabase)
            results.append(result)
            print()
        except Exception as e:
            print(f"   ❌ Test error: {e}")
            print()
            results.append(False)

    # Summary
    print("="*80)
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"✅ All tests passed ({passed}/{total})")
        print()
        print("Database is ready for use!")
        print("Next step: Configure dashboard in config/settings.py")
    else:
        print(f"⚠️  Some tests failed ({passed}/{total} passed)")
        print()
        print("Troubleshooting:")
        print("- If tables don't exist: Run setup_supabase_schema.sql in Supabase SQL Editor")
        print("- If no data exists: Run python scripts/migrate_existing_data.py")
        print("- Check credentials in .env file")

    print("="*80)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
