#!/bin/bash
# Run All MPC Diagnostic Tests
# Zentus - ERCOT Battery Revenue Dashboard

set -e  # Exit on error

echo "=========================================="
echo "MPC DIAGNOSTIC TEST SUITE"
echo "=========================================="
echo ""

# Enable diagnostic logging
export MPC_DIAGNOSTICS=true

# Change to ERCOT-Dashboard directory
cd "$(dirname "$0")/.."

echo "Current directory: $(pwd)"
echo ""

# Test 1: Price Pattern Analysis
echo "=========================================="
echo "TEST 1: Price Pattern Analysis (H1)"
echo "=========================================="
python3 tests/analyze_ercot_prices.py
echo ""

# Test 2: Detailed Strategy Comparison
echo "=========================================="
echo "TEST 2: Detailed Strategy Comparison"
echo "=========================================="
python3 tests/compare_strategies_detailed.py
echo ""

# Test 3: Comprehensive MPC Investigation
echo "=========================================="
echo "TEST 3: Comprehensive MPC Investigation"
echo "=========================================="
python3 tests/investigate_mpc_comprehensive.py
echo ""

echo "=========================================="
echo "ALL TESTS COMPLETE"
echo "=========================================="
