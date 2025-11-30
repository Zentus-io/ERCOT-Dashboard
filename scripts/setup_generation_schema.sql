-- Create table for ERCOT generation data (Wind + Solar)
-- Stores regional or nodal generation data linked to settlement points

CREATE TABLE IF NOT EXISTS ercot_generation (
    timestamp TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    settlement_point TEXT NOT NULL,
    fuel_type TEXT NOT NULL, -- 'Solar', 'Wind'
    gen_mw FLOAT,
    forecast_mw FLOAT, -- STWPF / STPPF
    potential_mw FLOAT, -- WGRPP / PVGRPP / HSL
    region TEXT, -- e.g., 'PVGR_North', 'WGR_West', 'System'
    
    -- Composite primary key to ensure uniqueness
    PRIMARY KEY (timestamp, settlement_point, fuel_type)
);

-- Create index for fast querying by settlement point and time
CREATE INDEX IF NOT EXISTS idx_ercot_generation_sp_time 
ON ercot_generation (settlement_point, timestamp);

-- Enable Row Level Security (RLS)
ALTER TABLE ercot_generation ENABLE ROW LEVEL SECURITY;

-- Drop existing policies to ensure clean state
DROP POLICY IF EXISTS "Allow public read access" ON ercot_generation;
DROP POLICY IF EXISTS "Allow authenticated full access" ON ercot_generation;
DROP POLICY IF EXISTS "Allow anon full access" ON ercot_generation;

-- Create policy to allow read access to everyone (public)
CREATE POLICY "Allow public read access" 
ON ercot_generation FOR SELECT 
TO anon, authenticated 
USING (true);

-- Create policy to allow insert/update/delete to authenticated users
CREATE POLICY "Allow authenticated full access" 
ON ercot_generation FOR ALL 
TO authenticated 
USING (true) 
WITH CHECK (true);

-- Create policy to allow insert/update/delete to anon users (if using anon key)
-- WARNING: This is for development convenience. In production, restrict this.
CREATE POLICY "Allow anon full access" 
ON ercot_generation FOR ALL 
TO anon 
USING (true) 
WITH CHECK (true);

-- Grant permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON ercot_generation TO anon, authenticated;
GRANT ALL ON ercot_generation TO service_role;
