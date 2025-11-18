-- ERCOT Battery Storage Revenue Dashboard
-- Supabase Database Schema
-- Zentus - 2025

-- Drop existing objects if they exist (for clean reinstall)
DROP MATERIALIZED VIEW IF EXISTS ercot_prices_merged CASCADE;
DROP TABLE IF EXISTS ercot_prices CASCADE;
DROP TABLE IF EXISTS eia_batteries CASCADE;

-- Main price table for ERCOT market data
CREATE TABLE ercot_prices (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    interval_start TIMESTAMPTZ NOT NULL,
    interval_end TIMESTAMPTZ NOT NULL,
    location TEXT NOT NULL,
    location_type TEXT NOT NULL,  -- 'Resource Node', 'Hub', 'Load Zone'
    market TEXT NOT NULL,          -- 'DAM' (Day-Ahead) or 'RTM' (Real-Time)
    price_mwh DECIMAL(10,2) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Unique constraint to prevent duplicate data
    CONSTRAINT unique_price_record UNIQUE (timestamp, interval_start, location, market)
);

-- Indexes for fast querying
-- Primary query pattern: filter by location + market + time range
CREATE INDEX idx_location_market_time ON ercot_prices (location, market, timestamp DESC);

-- Time-based queries (for date range filtering)
CREATE INDEX idx_timestamp ON ercot_prices (timestamp DESC);

-- Market-specific queries
CREATE INDEX idx_market ON ercot_prices (market);

-- Location lookup (for dropdown population)
CREATE INDEX idx_location ON ercot_prices (location);

-- Comment on table
COMMENT ON TABLE ercot_prices IS 'ERCOT settlement point prices from Day-Ahead and Real-Time markets. Data sourced from ERCOT API via gridstatus library.';

-- EIA-860 Battery Reference Data
CREATE TABLE eia_batteries (
    id SERIAL PRIMARY KEY,
    nameplate_power_mw DECIMAL(10,2),
    nameplate_energy_mwh DECIMAL(10,2),
    duration_hours DECIMAL(10,2),
    state TEXT,
    county TEXT,
    operator TEXT,
    use_arbitrage BOOLEAN,
    use_frequency_regulation BOOLEAN,
    use_ramping_reserve BOOLEAN,
    use_spinning_reserve BOOLEAN,
    online_year INTEGER,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE eia_batteries IS 'Battery storage systems in Texas from EIA-860 data. Used for market context and validation of battery specifications.';

-- Create materialized view for merged DAM + RTM prices
-- This pre-joins the data and calculates derived metrics for dashboard performance
CREATE MATERIALIZED VIEW ercot_prices_merged AS
SELECT
    d.timestamp,
    d.location AS node,
    d.price_mwh AS price_mwh_da,
    r.price_mwh AS price_mwh_rt,
    (r.price_mwh - d.price_mwh) AS forecast_error,
    ABS(r.price_mwh - d.price_mwh) AS price_spread,
    CASE
        WHEN ABS(r.price_mwh - d.price_mwh) > 10 THEN TRUE
        ELSE FALSE
    END AS extreme_event
FROM ercot_prices d
INNER JOIN ercot_prices r
    ON d.location = r.location
    AND d.timestamp = r.timestamp
WHERE d.market = 'DAM'
    AND r.market = 'RTM';

-- Index on materialized view for fast filtering
CREATE INDEX idx_merged_node_time ON ercot_prices_merged (node, timestamp DESC);
CREATE INDEX idx_merged_timestamp ON ercot_prices_merged (timestamp DESC);

COMMENT ON MATERIALIZED VIEW ercot_prices_merged IS 'Pre-joined view of DAM and RTM prices with calculated metrics. Refresh periodically after new data ingestion.';

-- Function to refresh the materialized view
CREATE OR REPLACE FUNCTION refresh_prices_merged()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY ercot_prices_merged;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION refresh_prices_merged() IS 'Refreshes the ercot_prices_merged materialized view. Call after bulk data ingestion.';

-- Enable Row Level Security (RLS) for security
ALTER TABLE ercot_prices ENABLE ROW LEVEL SECURITY;
ALTER TABLE eia_batteries ENABLE ROW LEVEL SECURITY;

-- Create policies to allow read access (adjust as needed for your use case)
CREATE POLICY "Enable read access for all users" ON ercot_prices
    FOR SELECT
    USING (true);

CREATE POLICY "Enable read access for all users" ON eia_batteries
    FOR SELECT
    USING (true);

-- If you need write access for the anon key (for data ingestion scripts)
CREATE POLICY "Enable insert for authenticated users" ON ercot_prices
    FOR INSERT
    WITH CHECK (true);

-- Helper view to get available date ranges per location
CREATE OR REPLACE VIEW available_data_summary AS
SELECT
    location,
    market,
    MIN(timestamp) AS earliest_data,
    MAX(timestamp) AS latest_data,
    COUNT(*) AS record_count,
    MAX(created_at) AS last_updated
FROM ercot_prices
GROUP BY location, market
ORDER BY location, market;

COMMENT ON VIEW available_data_summary IS 'Summary of available data coverage by location and market. Used for dashboard data freshness indicators.';

-- Grant permissions (adjust based on your Supabase setup)
-- GRANT ALL ON ALL TABLES IN SCHEMA public TO authenticated;
-- GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO authenticated;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO authenticated;

-- Schema setup complete
-- Next steps:
-- 1. Run this SQL in Supabase SQL Editor
-- 2. Use scripts/migrate_existing_data.py to import CSV data
-- 3. Use scripts/fetch_ercot_data.py to fetch new data from ERCOT API
