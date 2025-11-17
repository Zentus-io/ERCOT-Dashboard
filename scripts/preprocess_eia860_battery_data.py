"""
EIA-860 Battery Storage Data Preprocessing Script

Loads, cleans, and optimizes EIA Form 860 Energy Storage data for Texas (ERCOT) batteries.
Outputs an optimized Parquet file for fast loading in the dashboard.

Author: Zentus
Date: November 2025
"""

import polars as pl
import pandas as pd
from pathlib import Path
import sys

def main():
    """Main preprocessing pipeline for EIA-860 battery data."""

    # Define paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    input_file = project_dir / 'data' / 'eia8602024' / '3_4_Energy_Storage_Y2024.xlsx'
    output_dir = project_dir / 'data' / 'processed'
    output_file = output_dir / 'ercot_batteries.parquet'

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EIA-860 Battery Storage Data Preprocessing")
    print("=" * 80)
    print(f"\nInput:  {input_file}")
    print(f"Output: {output_file}")

    # Step 1: Load Excel file using pandas (polars Excel support is limited)
    print("\n[1/6] Loading Excel file (Operable sheet)...")
    try:
        df_pandas = pd.read_excel(
            input_file,
            sheet_name='Operable',
            header=1,  # Skip title row
            engine='openpyxl'
        )
        print(f"   ✓ Loaded {len(df_pandas):,} rows, {len(df_pandas.columns)} columns")
    except FileNotFoundError:
        print(f"   ✗ Error: Input file not found at {input_file}")
        sys.exit(1)
    except Exception as e:
        print(f"   ✗ Error loading Excel file: {e}")
        sys.exit(1)

    # Step 2: Clean and convert to Polars for efficient processing
    print("\n[2/6] Cleaning data and converting to Polars...")

    # Replace spaces and empty strings with NaN across all columns
    df_pandas = df_pandas.replace(r'^\s*$', pd.NA, regex=True)

    # Convert numeric-looking columns to proper numeric types
    numeric_cols = [
        'Nameplate Capacity (MW)', 'Summer Capacity (MW)', 'Winter Capacity (MW)',
        'Nameplate Energy Capacity (MWh)', 'Maximum Charge Rate (MW)',
        'Maximum Discharge Rate (MW)', 'Operating Month', 'Operating Year',
        'Nameplate Reactive Power Rating', 'Utility ID', 'Plant Code', 'Sector'
    ]

    for col in numeric_cols:
        if col in df_pandas.columns:
            df_pandas[col] = pd.to_numeric(df_pandas[col], errors='coerce')

    # Convert to Polars with string inference disabled for safety
    try:
        df = pl.from_pandas(df_pandas, include_index=False)
    except Exception as e:
        # Fallback: save to CSV and load with Polars (more robust)
        print(f"   ⚠ Direct conversion failed, using CSV intermediate: {e}")
        temp_csv = project_dir / 'data' / 'temp_conversion.csv'
        df_pandas.to_csv(temp_csv, index=False)
        df = pl.read_csv(temp_csv, infer_schema_length=10000)
        temp_csv.unlink()  # Delete temp file

    print(f"   ✓ Cleaned and converted to Polars DataFrame")

    # Step 3: Filter for Texas (ERCOT) batteries only
    print("\n[3/6] Filtering for Texas batteries...")
    df_texas = df.filter(pl.col('State') == 'TX')
    print(f"   ✓ Filtered {len(df_texas):,} Texas batteries (from {len(df):,} total)")

    # Step 4: Select and rename important columns
    print("\n[4/6] Selecting important columns...")

    df_clean = df_texas.select([
        # Identification
        pl.col('Plant Code').alias('plant_code'),
        pl.col('Plant Name').alias('plant_name'),
        pl.col('State').alias('state'),
        pl.col('County').alias('county'),
        pl.col('Generator ID').alias('generator_id'),

        # Technical specifications
        pl.col('Technology').alias('technology'),
        pl.col('Prime Mover').alias('prime_mover'),
        pl.col('Storage Technology 1').alias('storage_technology'),
        pl.col('Status').alias('status'),

        # Capacity metrics (MW and MWh)
        pl.col('Nameplate Capacity (MW)').alias('nameplate_power_mw'),
        pl.col('Summer Capacity (MW)').alias('summer_power_mw'),
        pl.col('Winter Capacity (MW)').alias('winter_power_mw'),
        pl.col('Nameplate Energy Capacity (MWh)').alias('nameplate_energy_mwh'),
        pl.col('Maximum Charge Rate (MW)').alias('max_charge_mw'),
        pl.col('Maximum Discharge Rate (MW)').alias('max_discharge_mw'),

        # Operational info
        pl.col('Operating Month').alias('operating_month'),
        pl.col('Operating Year').alias('operating_year'),
        pl.col('Sector Name').alias('sector_name'),
        pl.col('Sector').alias('sector_code'),

        # Use cases (binary flags)
        pl.col('Arbitrage').alias('use_arbitrage'),
        pl.col('Frequency Regulation').alias('use_frequency_regulation'),
        pl.col('Load Following').alias('use_load_following'),
        pl.col('Ramping / Spinning Reserve').alias('use_ramping_reserve'),
        pl.col('Co-Located Renewable Firming').alias('use_renewable_firming'),
        pl.col('Transmission and Distribution Deferral').alias('use_td_deferral'),
        pl.col('System Peak Shaving').alias('use_peak_shaving'),
        pl.col('Load Management').alias('use_load_management'),

        # Coupling type
        pl.col('AC Coupled').alias('ac_coupled'),
        pl.col('DC Coupled').alias('dc_coupled'),
        pl.col('Independent').alias('independent')
    ])

    print(f"   ✓ Selected {len(df_clean.columns)} important columns")

    # Step 5: Calculate derived metrics and optimize data types
    print("\n[5/6] Calculating derived metrics and optimizing data types...")

    df_optimized = df_clean.with_columns([
        # Calculate storage duration (hours)
        (pl.col('nameplate_energy_mwh') / pl.col('nameplate_power_mw')).alias('duration_hours'),

        # Calculate energy-to-power ratio
        (pl.col('nameplate_energy_mwh') / pl.col('nameplate_power_mw')).alias('e_to_p_ratio'),

        # Create a primary use case flag (prioritize arbitrage)
        pl.when(pl.col('use_arbitrage') == 'Y')
          .then(pl.lit('Arbitrage'))
          .when(pl.col('use_frequency_regulation') == 'Y')
          .then(pl.lit('Frequency Regulation'))
          .when(pl.col('use_ramping_reserve') == 'Y')
          .then(pl.lit('Ramping Reserve'))
          .otherwise(pl.lit('Other'))
          .alias('primary_use_case'),

        # Convert Operating Year to integer (if not null)
        pl.col('operating_year').cast(pl.Int32, strict=False)
    ])

    # Sort by capacity (largest first)
    df_optimized = df_optimized.sort('nameplate_power_mw', descending=True)

    print(f"   ✓ Added 3 derived columns")
    print(f"   ✓ Optimized data types")

    # Step 6: Save as Parquet
    print("\n[6/6] Saving optimized Parquet file...")
    try:
        df_optimized.write_parquet(
            output_file,
            compression='zstd',  # Fast compression with good ratio
            statistics=True,     # Enable statistics for query optimization
            use_pyarrow=True     # Use PyArrow for compatibility
        )

        # Get file sizes
        input_size_mb = input_file.stat().st_size / (1024 * 1024)
        output_size_mb = output_file.stat().st_size / (1024 * 1024)
        compression_ratio = (1 - output_size_mb / input_size_mb) * 100

        print(f"   ✓ Saved to {output_file}")
        print(f"   ✓ Input size:  {input_size_mb:.2f} MB")
        print(f"   ✓ Output size: {output_size_mb:.2f} MB")
        print(f"   ✓ Compression: {compression_ratio:.1f}% reduction")
    except Exception as e:
        print(f"   ✗ Error saving Parquet file: {e}")
        sys.exit(1)

    # Summary statistics
    print("\n" + "=" * 80)
    print("DATA SUMMARY")
    print("=" * 80)
    print(f"Total Texas batteries:        {len(df_optimized):,}")
    print(f"Total installed power:        {df_optimized['nameplate_power_mw'].sum():,.1f} MW")
    print(f"Total installed energy:       {df_optimized['nameplate_energy_mwh'].sum():,.1f} MWh")
    print(f"Average system size:          {df_optimized['nameplate_power_mw'].mean():.1f} MW")
    print(f"Median system size:           {df_optimized['nameplate_power_mw'].median():.1f} MW")
    print(f"Average duration:             {df_optimized['duration_hours'].mean():.2f} hours")
    print(f"Median duration:              {df_optimized['duration_hours'].median():.2f} hours")

    # Use case breakdown
    print("\nPrimary Use Cases:")
    use_case_counts = df_optimized.group_by('primary_use_case').agg(pl.len().alias('count'))
    for row in use_case_counts.iter_rows():
        use_case, count = row
        percentage = (count / len(df_optimized)) * 100
        print(f"  {use_case:25s} {count:3d} systems ({percentage:5.1f}%)")

    # Technology breakdown
    print("\nStorage Technologies:")
    tech_counts = df_optimized.group_by('storage_technology').agg(pl.len().alias('count')).sort('count', descending=True)
    for row in tech_counts.iter_rows():
        tech, count = row
        if tech:  # Skip null values
            percentage = (count / len(df_optimized)) * 100
            print(f"  {tech:25s} {count:3d} systems ({percentage:5.1f}%)")

    print("\n" + "=" * 80)
    print("✓ Preprocessing complete!")
    print("=" * 80)
    print(f"\nOutput file: {output_file}")
    print(f"Load in Python with: df = pl.read_parquet('{output_file.name}')")
    print()

if __name__ == "__main__":
    main()
