-- ERCOT Battery Storage Revenue Dashboard - Optimized Schema V2
-- Zentus - 2025

-- This schema is for a fresh database setup.
-- If you have existing data, run 'migrate_schema_to_v2.sql' instead of this.

DROP TABLE IF EXISTS ercot_prices CASCADE;
DROP TABLE IF EXISTS eia_batteries CASCADE;
DROP FUNCTION IF EXISTS get_daily_summary(date, date);
DROP FUNCTION IF EXISTS get_date_range();

-- Optimized price table for ERCOT market data
CREATE TABLE ercot_prices (
    timestamp TIMESTAMP(0) WITHOUT TIME ZONE NOT NULL,
    settlement_point TEXT NOT NULL,
    market TEXT NOT NULL,          -- 'DAM' (Day-Ahead) or 'RTM' (Real-Time)
    price_mwh DECIMAL(10, 2) NOT NULL,

    -- Composite primary key to enforce uniqueness and save space
    PRIMARY KEY (timestamp, settlement_point, market)
);

-- Indexes for fast querying
CREATE INDEX idx_settlement_point_market_time ON ercot_prices (settlement_point, market, timestamp DESC);
CREATE INDEX idx_timestamp_market ON ercot_prices (timestamp, market);  -- For gap detection performance

-- Comment on table
COMMENT ON TABLE ercot_prices IS 'ERCOT settlement point prices. Optimized for space. Fetched from ERCOT API via gridstatus.';

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

COMMENT ON TABLE eia_batteries IS 'Battery storage systems in Texas from EIA-860 data. Used for market context.';

-- Function to get daily record counts for gap analysis
CREATE OR REPLACE FUNCTION get_daily_summary(
    p_start_date date,
    p_end_date date
)
RETURNS TABLE(day timestamp, market text, record_count bigint) AS $$
BEGIN
    RETURN QUERY
    SELECT
        date_trunc('day', T.timestamp)::timestamp as day,
        T.market,
        count(*) as record_count
    FROM ercot_prices AS T
    WHERE T.timestamp >= p_start_date AND T.timestamp < p_end_date + interval '1 day'
    GROUP BY 1, 2;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_daily_summary(date, date) IS 'Returns the daily record count per market for a given date range.';

-- Function to get the earliest and latest dates in the database
CREATE OR REPLACE FUNCTION get_date_range()
RETURNS TABLE(min_date timestamp, max_date timestamp) AS $$
BEGIN
    RETURN QUERY
    SELECT
        MIN(timestamp) as min_date,
        MAX(timestamp) as max_date
    FROM ercot_prices;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_date_range() IS 'Returns the minimum and maximum timestamp in the ercot_prices table.';


-- Enable Row Level Security (RLS) for security
ALTER TABLE ercot_prices ENABLE ROW LEVEL SECURITY;
ALTER TABLE eia_batteries ENABLE ROW LEVEL SECURITY;

-- Create permissive policies for data ingestion and dashboard reading.
CREATE POLICY "Enable all actions for data ingestion" ON ercot_prices
    FOR ALL
    USING (true)
    WITH CHECK (true);

CREATE POLICY "Enable read access for all users" ON eia_batteries
    FOR SELECT
    USING (true);

-- Engie/Broad Reach Power Assets
CREATE TABLE engie_storage_assets (
    id SERIAL PRIMARY KEY,
    plant_name TEXT,
    settlement_point TEXT NOT NULL,
    nameplate_power_mw DECIMAL(10,2),
    nameplate_energy_mwh DECIMAL(10,2),
    duration_hours DECIMAL(10,2),
    county TEXT,
    inferred_zone TEXT,
    status_type TEXT,
    operating_year INTEGER,
    resource_name TEXT,
    asset_id TEXT,
    hsl DECIMAL(10,2),
    lsl DECIMAL(10,2),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE engie_storage_assets IS 'Engie and Broad Reach Power battery assets in ERCOT.';

ALTER TABLE engie_storage_assets ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Enable read access for all users" ON engie_storage_assets
    FOR SELECT
    USING (true);

CREATE POLICY "Enable all actions for data ingestion" ON engie_storage_assets
    FOR ALL
    USING (true)
    WITH CHECK (true);