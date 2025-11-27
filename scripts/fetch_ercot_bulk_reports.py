"""
ERCOT Bulk Report Downloader (Official API)
Zentus - ERCOT Battery Revenue Dashboard

Purpose:
    Downloads raw bulk data reports (ZIP/CSV) directly from the ERCOT Public API.
    Useful for fetching historical archives or specific report types not supported
    by the `gridstatus` library. Data is saved locally as Parquet/CSV.

Usage:
    python scripts/fetch_ercot_bulk_reports.py --report RTM --start 2025-01-01
    python scripts/fetch_ercot_bulk_reports.py --help

Arguments:
    --report (str): Report type(s) to fetch (RTM, RTM_HISTORICAL, DAM, DAM_AS).
    --start (str): Start date (YYYY-MM-DD).
    --end (str): End date (YYYY-MM-DD). Default: yesterday.
    --output (str): Output directory.
    --format (str): Output format (csv/parquet).

Examples:
    # Download RTM Settlement Point Prices for 2025
    python scripts/fetch_ercot_bulk_reports.py --report RTM --start 2025-01-01

    # Download DAM Prices for a specific range
    python scripts/fetch_ercot_bulk_reports.py --report DAM --start 2025-01-01 --end 2025-01-31
"""

import argparse
import io
import os
import sys
import time
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl
import requests
from dotenv import load_dotenv
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

# ERCOT API endpoints
AUTH_URL = (
    "https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com/"
    "B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token"
)
API_BASE_URL = "https://api.ercot.com/api/public-reports"

# Report configurations
REPORTS = {
    "RTM": {
        "emil_id": "NP6-905-CD",
        "name": "Settlement Point Prices at Resource Nodes, Hubs and Load Zones",
        "description": "Real-time 15-minute settlement point prices",
        "output_dir": "RTM_Prices",
        "bulk_limit": 200,  # Reduced to avoid rate limiting (was 500)
        "file_pattern": "SPPHLZNP6905",  # Pattern to identify already downloaded files
    },
    "RTM_HISTORICAL": {
        "emil_id": "NP6-785-ER",
        "name": "Historical RTM Load Zone and Hub Prices",
        "description": "Weekly aggregated historical RTM prices (xlsx)",
        "output_dir": "RTM_Historical",
        "bulk_limit": 100,
        "file_pattern": "HistoricalRTM",
    },
    "DAM": {
        "emil_id": "NP4-190-CD",
        "name": "DAM Settlement Point Prices",
        "description": "Day-ahead hourly settlement point prices",
        "output_dir": "DAM_Prices",
        "bulk_limit": 500,
        "file_pattern": "DAMSPNP4190",
    },
    "DAM_AS": {
        "emil_id": "NP4-188-CD",
        "name": "DAM Clearing Prices for Capacity",
        "description": "Day-ahead Ancillary Services market clearing prices (RegUp, RegDn, RRS, ECRS, NonSpin)",
        "output_dir": "DAM_AS_Prices",
        "bulk_limit": 500,
        "file_pattern": "DAMASNP4188",
    },
}

# Rate limiting - ERCOT API has strict limits, be conservative
REQUEST_DELAY = 2.0  # seconds between API calls (increased to avoid 429s)
REQUEST_DELAY_ON_429 = 30  # seconds to wait after a 429 error
MAX_RETRIES = 5  # Increased retries for rate limiting
TOKEN_EXPIRY_BUFFER = 300  # Refresh token 5 min before expiry


# ============================================================================
# ERCOT API CLIENT
# ============================================================================

class ERCOTAPIClient:
    """Client for interacting with the ERCOT Public API."""

    def __init__(self, username: str, password: str, subscription_key: str):
        self.username = username
        self.password = password
        self.subscription_key = subscription_key
        self.id_token: Optional[str] = None
        self.token_expiry: Optional[datetime] = None

    def authenticate(self) -> bool:
        """Obtain an ID token from ERCOT B2C."""
        print("   Authenticating with ERCOT API...")

        # Use form data (not query params) for authentication
        data = {
            "username": self.username,
            "password": self.password,
            "grant_type": "password",
            "scope": "openid fec253ea-0d06-4272-a5e6-b478baeecd70 offline_access",
            "client_id": "fec253ea-0d06-4272-a5e6-b478baeecd70",
            "response_type": "id_token",
        }

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
        }

        try:
            response = requests.post(AUTH_URL, data=data, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()

            self.id_token = data.get("id_token") or data.get("access_token")
            if not self.id_token:
                print(f"   Error: No token in response. Keys: {list(data.keys())}")
                return False

            # Token expires in 1 hour
            expires_in = int(data.get("expires_in", 3600))
            self.token_expiry = datetime.now() + timedelta(seconds=expires_in)

            print(f"   Authentication successful. Token expires at {self.token_expiry.strftime('%H:%M:%S')}")
            return True

        except requests.exceptions.RequestException as e:
            print(f"   Authentication failed: {e}")
            return False

    def _ensure_authenticated(self) -> bool:
        """Ensure we have a valid token, refreshing if necessary."""
        if self.id_token is None:
            return self.authenticate()

        if self.token_expiry and datetime.now() > (self.token_expiry - timedelta(seconds=TOKEN_EXPIRY_BUFFER)):
            print("   Token expiring soon, refreshing...")
            return self.authenticate()

        return True

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        return {
            "Authorization": f"Bearer {self.id_token}",
            "Ocp-Apim-Subscription-Key": self.subscription_key,
        }

    def list_archives(
        self,
        emil_id: str,
        start_date: date,
        end_date: date,
        page: int = 1,
        page_size: int = 1000,
    ) -> Dict[str, Any]:
        """List available archive files for a report within a date range."""
        if not self._ensure_authenticated():
            return {"archives": [], "total": 0}

        url = f"{API_BASE_URL}/archive/{emil_id}"
        params = {
            "postDatetimeFrom": start_date.strftime("%Y-%m-%d"),
            "postDatetimeTo": (end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
            "page": page,
            "size": page_size,
        }

        for retry in range(MAX_RETRIES):
            try:
                response = requests.get(url, headers=self._get_headers(), params=params, timeout=60)

                # Handle rate limiting specifically
                if response.status_code == 429:
                    wait_time = REQUEST_DELAY_ON_429 * (retry + 1)
                    print(f"   Rate limited (429). Waiting {wait_time}s before retry {retry + 1}/{MAX_RETRIES}...")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                data = response.json()

                return {
                    "archives": data.get("archives", []),
                    "total": data.get("_meta", {}).get("totalRecords", 0),
                    "total_pages": data.get("_meta", {}).get("totalPages", 1),
                    "current_page": data.get("_meta", {}).get("currentPage", 1),
                }

            except requests.exceptions.RequestException as e:  # pylint: disable=broad-except
                if retry < MAX_RETRIES - 1:
                    wait_time = REQUEST_DELAY_ON_429 * (retry + 1)
                    print(f"   Error listing archives: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"   Error listing archives after {MAX_RETRIES} retries: {e}")
                    return {"archives": [], "total": 0}

        return {"archives": [], "total": 0}

    def download_single_archive(self, emil_id: str, doc_id: int) -> Optional[bytes]:
        """Download a single archive file by document ID."""
        if not self._ensure_authenticated():
            return None

        url = f"{API_BASE_URL}/archive/{emil_id}"
        params = {"download": doc_id}

        try:
            response = requests.get(
                url, headers=self._get_headers(), params=params, timeout=120
            )
            response.raise_for_status()
            return response.content

        except requests.exceptions.RequestException as e:  # pylint: disable=broad-except
            print(f"   Error downloading {doc_id}: {e}")
            return None

    def download_bulk_archives(
        self, emil_id: str, doc_ids: List[int], retry_count: int = 0
    ) -> Optional[bytes]:
        """Download multiple archive files in a single request (returns ZIP)."""
        if not self._ensure_authenticated():
            return None

        url = f"{API_BASE_URL}/archive/{emil_id}/download"
        payload = {"docIds": doc_ids}

        try:
            response = requests.post(
                url, headers=self._get_headers(), json=payload, timeout=300
            )

            # Handle rate limiting specifically
            if response.status_code == 429:
                if retry_count < MAX_RETRIES:
                    wait_time = REQUEST_DELAY_ON_429 * (retry_count + 1)
                    print(f"   Rate limited (429). Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    return self.download_bulk_archives(emil_id, doc_ids, retry_count + 1)
                print(f"   Rate limit exceeded after {MAX_RETRIES} retries")
                return None

            response.raise_for_status()
            return response.content

        except requests.exceptions.RequestException as e:  # pylint: disable=broad-except
            print(f"   Error in bulk download: {e}")
            return None


# ============================================================================
# DATA PROCESSING
# ============================================================================

def extract_and_save_files(
    zip_content: bytes,
    output_dir: Path,
    extract_csv: bool = True,
) -> List[str]:
    """Extract files from a ZIP archive and save to output directory."""
    saved_files = []

    try:
        with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
            for filename in zf.namelist():
                # Skip directories
                if filename.endswith('/'):
                    continue

                # Extract file
                content = zf.read(filename)

                # Handle nested zips (ERCOT sometimes double-zips)
                if filename.endswith('.zip'):
                    nested_saved = extract_and_save_files(content, output_dir, extract_csv)
                    saved_files.extend(nested_saved)
                else:
                    # Save directly
                    output_path = output_dir / Path(filename).name
                    with open(output_path, 'wb') as f:
                        f.write(content)
                    saved_files.append(str(output_path))

    except zipfile.BadZipFile:
        # Not a zip, might be raw CSV - save directly
        # Generate a filename based on timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"data_{timestamp}.csv"
        with open(output_path, 'wb') as f:
            f.write(zip_content)
        saved_files.append(str(output_path))

    return saved_files


def consolidate_csv_files(
    input_dir: Path,
    output_file: Path,
    date_filter: Optional[tuple] = None,
    output_format: str = "csv",
) -> int:
    """Consolidate multiple CSV files into a single file using Polars (fast!).

    Args:
        input_dir: Directory containing CSV files
        output_file: Output file path (extension will be adjusted based on format)
        date_filter: Optional date range filter (not implemented yet)
        output_format: "csv" or "parquet"
    """
    # Exclude any existing consolidated files from the list
    csv_files = [f for f in input_dir.glob("*.csv") if "_consolidated_" not in f.name]

    if not csv_files:
        print(f"   No CSV files found in {input_dir}")
        return 0

    # Delete old consolidated files first (both csv and parquet)
    for pattern in ["*_consolidated_*.csv", "*_consolidated_*.parquet"]:
        old_consolidated = list(input_dir.glob(pattern))
        for old_file in old_consolidated:
            print(f"   Removing old consolidated file: {old_file.name}")
            old_file.unlink()

    print(f"   Consolidating {len(csv_files)} CSV files with Polars...")

    # Schema overrides to ensure consistent types across all files
    # SettlementPointPrice can be integer or float depending on the file
    schema_overrides = {
        "SettlementPointPrice": pl.Float64,
        "SettlementPointName": pl.Utf8,
        "SettlementPoint": pl.Utf8,
    }

    dfs = []
    for csv_file in tqdm(csv_files, desc="Reading CSVs", unit="file"):
        try:
            df = pl.read_csv(csv_file, schema_overrides=schema_overrides, infer_schema_length=1000)
            dfs.append(df)
        except Exception as e:  # pylint: disable=broad-except
            print(f"   Warning: Could not read {csv_file.name}: {e}")

    if not dfs:
        return 0

    # Concatenate all dataframes
    combined = pl.concat(dfs)

    # Remove duplicates if any
    if 'DeliveryDate' in combined.columns and 'SettlementPoint' in combined.columns:
        before = len(combined)
        subset_cols = (
            ['DeliveryDate', 'HourEnding', 'SettlementPoint']
            if 'HourEnding' in combined.columns
            else ['DeliveryDate', 'DeliveryHour', 'DeliveryInterval', 'SettlementPointName']
        )
        combined = combined.unique(subset=subset_cols, keep='last')
        after = len(combined)
        if before != after:
            print(f"   Removed {before - after} duplicate records")

    # Drop source_file column if it exists from previous runs
    if 'source_file' in combined.columns:
        combined = combined.drop('source_file')

    # Save in requested format
    if output_format == "parquet":
        output_file = output_file.with_suffix(".parquet")
        combined.write_parquet(output_file, compression="snappy")
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"   Saved consolidated file: {output_file} ({len(combined):,} records, {file_size_mb:.1f} MB)")
    else:
        combined.write_csv(output_file)
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"   Saved consolidated file: {output_file} ({len(combined):,} records, {file_size_mb:.1f} MB)")

    return len(combined)


# ============================================================================
# MAIN FETCHER
# ============================================================================

def get_existing_files(output_dir: Path) -> set:
    """Get set of existing file identifiers in the output directory."""
    existing = set()
    for ext in ["*.csv", "*.zip", "*.xlsx"]:
        for f in output_dir.glob(ext):
            # Extract date/time identifier from filename
            # e.g., "cdr.00012331.0000000000000000.20250723.123216.DAMSPNP4190.csv"
            # We use the full filename (without extension) as identifier
            existing.add(f.stem)
    return existing


def fetch_report(
    client: ERCOTAPIClient,
    report_key: str,
    start_date: date,
    end_date: date,
    output_base: Path,
    consolidate: bool = True,
    skip_existing: bool = True,
    output_format: str = "csv",
) -> Dict[str, Any]:
    """Fetch all archives for a report within the date range."""

    report = REPORTS[report_key]
    emil_id = report["emil_id"]
    output_dir = output_base / report["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"Fetching: {report['name']}")
    print(f"EMIL ID: {emil_id}")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Output: {output_dir}")
    print(f"{'=' * 70}")

    # Check for existing files if skip_existing is enabled
    existing_files = set()
    if skip_existing:
        existing_files = get_existing_files(output_dir)
        if existing_files:
            print(f"\n   Found {len(existing_files)} existing files in output directory")

    # Step 1: List all available archives
    print("\n1. Listing available archives...")
    all_archives = []
    page = 1

    while True:
        result = client.list_archives(emil_id, start_date, end_date, page=page)
        archives = result.get("archives", [])

        if not archives:
            break

        all_archives.extend(archives)
        total_pages = result.get("total_pages", 1)

        print(
            f"   Page {page}/{total_pages}: Found {len(archives)} archives (Total: {len(all_archives)})")

        if page >= total_pages:
            break

        page += 1
        time.sleep(REQUEST_DELAY)

    if not all_archives:
        print("   No archives found for the specified date range.")
        return {"success": False, "files_downloaded": 0, "records": 0}

    print(f"\n   Total archives found: {len(all_archives)}")

    # Filter out already downloaded archives
    if skip_existing and existing_files:
        original_count = len(all_archives)
        # Filter archives whose friendlyName (without extension) is not already downloaded
        all_archives = [
            a for a in all_archives
            if Path(a.get("friendlyName", "")).stem not in existing_files
        ]
        skipped = original_count - len(all_archives)
        if skipped > 0:
            print(f"   Skipping {skipped} already downloaded archives")
            print(f"   Remaining to download: {len(all_archives)}")

    if not all_archives:
        print("   All archives already downloaded!")
        # Still consolidate if requested
        total_records = 0
        if consolidate:
            print(f"\n3. Consolidating existing CSV files to {output_format}...")
            consolidated_file = output_dir / \
                f"{report_key.lower()}_consolidated_{start_date}_{end_date}.csv"
            total_records = consolidate_csv_files(
                output_dir, consolidated_file, output_format=output_format)
        return {
            "success": True,
            "files_downloaded": 0,
            "records": total_records,
            "output_dir": str(output_dir),
            "skipped": True}

    # Step 2: Download archives in bulk batches
    print(f"\n2. Downloading archives (bulk limit: {report['bulk_limit']} per request)...")

    doc_ids = [archive["docId"] for archive in all_archives]
    bulk_limit = report["bulk_limit"]
    total_files_saved = []

    # Process in batches
    batches = [doc_ids[i:i + bulk_limit] for i in range(0, len(doc_ids), bulk_limit)]

    for batch_idx, batch_ids in enumerate(tqdm(batches, desc="Downloading batches", unit="batch")):
        batch_success = False
        for retry in range(MAX_RETRIES):
            try:
                zip_content = client.download_bulk_archives(emil_id, batch_ids)

                if zip_content:
                    saved = extract_and_save_files(zip_content, output_dir)
                    total_files_saved.extend(saved)
                    batch_success = True
                    break
                if retry < MAX_RETRIES - 1:
                    wait_time = REQUEST_DELAY_ON_429 * (retry + 1)
                    print(
                        f"   Retry {retry + 1}/{MAX_RETRIES} for batch {batch_idx + 1} in {wait_time}s...")
                    time.sleep(wait_time)

            except Exception as e:  # pylint: disable=broad-except
                print(f"   Error in batch {batch_idx + 1}: {e}")
                if retry < MAX_RETRIES - 1:
                    wait_time = REQUEST_DELAY_ON_429 * (retry + 1)
                    time.sleep(wait_time)

        if not batch_success:
            failed_count = len(batch_ids)
            print(f"   WARNING: Batch {batch_idx + 1} failed after {MAX_RETRIES} retries ({failed_count} files)")

        time.sleep(REQUEST_DELAY)

    print(f"\n   Files saved: {len(total_files_saved)}")

    # Step 3: Optionally consolidate CSVs
    total_records = 0
    if consolidate and total_files_saved:
        print(f"\n3. Consolidating CSV files to {output_format}...")
        consolidated_file = output_dir / \
            f"{report_key.lower()}_consolidated_{start_date}_{end_date}.csv"
        total_records = consolidate_csv_files(
            output_dir, consolidated_file, output_format=output_format)

    return {
        "success": True,
        "files_downloaded": len(total_files_saved),
        "records": total_records,
        "output_dir": str(output_dir),
    }


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fetch ERCOT data directly from the Public API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch RTM prices for 2025
  python fetch_ercot_api.py --report RTM --start 2025-01-01 --end 2025-11-24

  # Fetch historical RTM (weekly aggregated)
  python fetch_ercot_api.py --report RTM_HISTORICAL --start 2025-01-01

  # Fetch DAM prices and Ancillary Services prices
  python fetch_ercot_api.py --report DAM DAM_AS --start 2025-01-01 --end 2025-11-24

  # Fetch multiple reports
  python fetch_ercot_api.py --report RTM RTM_HISTORICAL DAM DAM_AS --start 2025-01-01

  # Force re-download (don't skip existing files)
  python fetch_ercot_api.py --report DAM --start 2025-01-01 --force

Available Reports:
  RTM            - Real-time 15-min settlement point prices (NP6-905-CD)
  RTM_HISTORICAL - Weekly aggregated historical RTM prices (NP6-785-ER)
  DAM            - Day-ahead hourly settlement point prices (NP4-190-CD)
  DAM_AS         - Day-ahead Ancillary Services clearing prices (NP4-188-CD)

Environment Variables (or use .env file):
  ERCOT_API_USERNAME     - Your ERCOT API Explorer username (email)
  ERCOT_API_PASSWORD     - Your ERCOT API Explorer password
  ERCOT_API_SUBSCRIPTION_KEY - Your subscription key from API Explorer
        """
    )

    parser.add_argument(
        "--report",
        type=str,
        nargs="+",
        choices=list(REPORTS.keys()),
        required=True,
        help="Report(s) to fetch",
    )
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD). Default: yesterday",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output base directory. Default: ERCOT-Dashboard/data/",
    )
    parser.add_argument(
        "--no-consolidate",
        action="store_true",
        help="Skip consolidating CSV files into a single file",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download of all files (don't skip existing)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "parquet"],
        default="parquet",
        help="Output format for consolidated file (default: parquet - 5-10x smaller)",
    )
    parser.add_argument(
        "--username",
        type=str,
        default=None,
        help="ERCOT API username (overrides env var)",
    )
    parser.add_argument(
        "--password",
        type=str,
        default=None,
        help="ERCOT API password (overrides env var)",
    )
    parser.add_argument(
        "--subscription-key",
        type=str,
        default=None,
        help="ERCOT API subscription key (overrides env var)",
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Get credentials
    username = args.username or os.getenv("ERCOT_API_USERNAME")
    password = args.password or os.getenv("ERCOT_API_PASSWORD")
    subscription_key = args.subscription_key or os.getenv("ERCOT_API_SUBSCRIPTION_KEY")

    if not all([username, password, subscription_key]):
        print("=" * 70)
        print("ERROR: ERCOT API credentials not found!")
        print("=" * 70)
        print("\nPlease set the following environment variables (or use .env file):")
        print("  ERCOT_API_USERNAME     - Your ERCOT API Explorer email")
        print("  ERCOT_API_PASSWORD     - Your ERCOT API Explorer password")
        print("  ERCOT_API_SUBSCRIPTION_KEY - Your subscription key")
        print("\nTo get credentials:")
        print("  1. Register at https://apiexplorer.ercot.com/")
        print("  2. Go to Products -> Subscribe to Public API")
        print("  3. Go to Profile -> Copy your Primary Key")
        return 1

    # At this point the values must be non-None; assert for the type checker/mypy
    assert username is not None and password is not None and subscription_key is not None

    # Parse dates
    start_date = datetime.strptime(args.start, "%Y-%m-%d").date()

    if args.end:
        end_date = datetime.strptime(args.end, "%Y-%m-%d").date()
    else:
        end_date = date.today() - timedelta(days=1)

    # Ensure end_date is not in the future
    if end_date > date.today() - timedelta(days=1):
        end_date = date.today() - timedelta(days=1)
        print(f"Note: End date capped at yesterday ({end_date})")

    # Set output directory
    if args.output:
        output_base = Path(args.output)
    else:
        # Default to ERCOT-Dashboard/data/ (consistent with repository structure)
        script_dir = Path(__file__).parent.parent
        output_base = script_dir / "data"

    output_base.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Zentus - ERCOT Direct API Data Fetcher (V1.0)")
    print("=" * 70)
    print(f"\nReports to fetch: {', '.join(args.report)}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Output directory: {output_base}")

    # Initialize API client
    print("\nInitializing ERCOT API client...")
    client = ERCOTAPIClient(username, password, subscription_key)

    if not client.authenticate():
        print("\nFailed to authenticate with ERCOT API.")
        print("Please check your credentials and try again.")
        return 1

    # Fetch each report
    results = {}
    for report_key in args.report:
        result = fetch_report(
            client,
            report_key,
            start_date,
            end_date,
            output_base,
            consolidate=not args.no_consolidate,
            skip_existing=not args.force,
            output_format=args.format,
        )
        results[report_key] = result

    # Print summary
    print("\n" + "=" * 70)
    print("FETCH SUMMARY")
    print("=" * 70)

    for report_key, result in results.items():
        status = "SUCCESS" if result["success"] else "FAILED"
        print(f"\n{report_key} ({REPORTS[report_key]['emil_id']}):")
        print(f"  Status: {status}")
        print(f"  Files downloaded: {result['files_downloaded']}")
        if result.get("records"):
            print(f"  Total records: {result['records']:,}")
        if result.get("output_dir"):
            print(f"  Output: {result['output_dir']}")

    print("\n" + "=" * 70)
    all_success = all(r["success"] for r in results.values())
    if all_success:
        print("All fetches completed successfully!")
    else:
        print("Some fetches failed. Check logs above.")
    print("=" * 70)

    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
