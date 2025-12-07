#!/bin/bash
# Test script for NEW SYNTAX Tau specs (v0.7.0-alpha)

set -e
cd "$(dirname "$0")"

# Set Tau binary path
TAU="${TAU:-/home/trevormoc/.cursor/worktrees/DeflationaryAgent/goi/tau-lang-latest/build-Release/tau}"

mkdir -p inputs outputs

echo "============================================"
echo "Testing NEW SYNTAX Tau Specs (v0.7.0-alpha)"
echo "============================================"
echo ""

# Test 1: Echo
echo "Test 1: Echo"
echo "1" > inputs/signal.in
echo "0" >> inputs/signal.in
echo "1" >> inputs/signal.in

rm -f outputs/*.out
cat 01_echo_new.tau | $TAU 2>&1 | grep -E "^[01]$" | head -5 || true
echo "Expected: 1 0 1"
echo "Output:   $(cat outputs/echo.out 2>/dev/null | tr '\n' ' ')"
echo ""

# Test 2: Toggle
echo "Test 2: Toggle"
echo "1" > inputs/signal.in
echo "0" >> inputs/signal.in
echo "1" >> inputs/signal.in
echo "0" >> inputs/signal.in

rm -f outputs/*.out
cat 04_toggle_new.tau | $TAU 2>&1 | grep -E "^[01]$" | head -5 || true
echo "Expected: 0 1 1 0 (starting with o1[0]=0, then toggle on 1, keep on 0)"
echo "Output:   $(cat outputs/toggle.out 2>/dev/null | tr '\n' ' ')"
echo ""

echo "============================================"
echo "Tests Complete"
echo "============================================"

