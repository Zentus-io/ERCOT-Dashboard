"""
Create Supabase Database Tables
Zentus - ERCOT Battery Revenue Dashboard

This script executes the SQL schema to create tables, indexes, and views in Supabase.
Run this once to set up the database structure.

Usage:
    python scripts/create_supabase_tables.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def setup_database():
    """Execute SQL schema to create database objects."""

    # Load environment variables
    load_dotenv()

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        print("❌ ERROR: Supabase credentials not found!")
        print("Please set SUPABASE_URL and SUPABASE_KEY in your .env file")
        print("See .env.example for template")
        sys.exit(1)

    # Initialize Supabase client
    print("🔌 Connecting to Supabase...")
    try:
        supabase: Client = create_client(supabase_url, supabase_key)
        print("✅ Connected successfully")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        sys.exit(1)

    # Read SQL schema file
    schema_file = Path(__file__).parent / "setup_supabase_schema.sql"

    if not schema_file.exists():
        print(f"❌ ERROR: Schema file not found at {schema_file}")
        sys.exit(1)

    print(f"📄 Reading schema from {schema_file.name}...")
    with open(schema_file, 'r') as f:
        sql_schema = f.read()

    # Note: Supabase Python client doesn't directly support raw SQL execution
    # You need to run the SQL manually in the Supabase SQL Editor
    # This script provides instructions

    print("\n" + "="*80)
    print("⚠️  IMPORTANT: Manual SQL Execution Required")
    print("="*80)
    print()
    print("The Supabase Python client does not support executing raw SQL DDL statements.")
    print("Please follow these steps:")
    print()
    print("1. Open your Supabase project dashboard:")
    print(f"   {supabase_url.replace('supabase.co', 'supabase.com/project')}")
    print()
    print("2. Navigate to: SQL Editor (left sidebar)")
    print()
    print("3. Click 'New Query'")
    print()
    print("4. Copy and paste the contents of:")
    print(f"   {schema_file.absolute()}")
    print()
    print("5. Click 'Run' to execute the schema")
    print()
    print("="*80)
    print()
    print("💡 Tip: After running the SQL, verify the tables were created:")
    print("   - Check 'Table Editor' in Supabase dashboard")
    print("   - You should see: ercot_prices, eia_batteries")
    print()
    print("Then run: python scripts/test_database_connection.py")
    print("="*80)

    return 0

if __name__ == "__main__":
    sys.exit(setup_database())
