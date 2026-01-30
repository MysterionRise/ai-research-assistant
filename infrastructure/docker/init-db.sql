-- ============================================================================
-- ARIA Research Assistant - Database Initialization
-- ============================================================================
-- Run automatically by PostgreSQL container on first start
-- ============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create test database for CI
CREATE DATABASE aria_test;
GRANT ALL PRIVILEGES ON DATABASE aria_test TO aria;

-- Connect to test database and enable extensions
\c aria_test
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Return to main database
\c aria_db

-- Create schemas
CREATE SCHEMA IF NOT EXISTS audit;
CREATE SCHEMA IF NOT EXISTS vectors;

-- Grant permissions
GRANT ALL ON SCHEMA audit TO aria;
GRANT ALL ON SCHEMA vectors TO aria;

-- Log completion
DO $$
BEGIN
    RAISE NOTICE 'ARIA database initialization completed successfully';
END $$;
