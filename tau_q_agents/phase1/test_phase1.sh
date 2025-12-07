#!/bin/bash
# Phase 1 Test Runner
# Tests foundation specs via Docker

set -e
cd "$(dirname "$0")"

mkdir -p outputs logs

# Wrapper to run a single test and capture output to logs/<name>.log
run_test() {
  local name="$1"
  local file="$2"
  local desc="$3"
  local expect="$4"

  echo "Test ${name}: ${desc}"
  if [ -n "$expect" ]; then
    echo "Expect: ${expect}"
  fi
  echo "---"

  # Allow command failure without aborting the entire script
  set +e
  docker run --rm \
    -v "$(pwd)":/work \
    -w /work \
    --entrypoint /tau-lang/build-Release/tau \
    tau:latest \
    "${file}" \
    >"logs/${name}.log" 2>&1
  status=$?
  set -e

  echo "Command exit status: ${status}"
  echo "Command output (first 40 lines):"
  head -40 "logs/${name}.log" || true
  echo ""
  echo "Output file:"
  cat "outputs/${name}.out" 2>/dev/null || echo "(no output file)"
  echo ""
}

echo "============================================"
echo "Phase 1: Foundation Specs"
echo "============================================"
echo ""

run_test "echo" "01_echo.tau" "Echo (o0[t] = i0[t])" "0,1,0,1,1,0"
run_test "negation" "02_negation.tau" "Negation (o0[t] = i0[t]')" "1,0,1,0,0,1"
run_test "delay" "03_delay.tau" "Delay (o0[t] = i0[t-1])" "?,0,1,0,1,1"
run_test "toggle" "04_toggle.tau" "Toggle (o0[t] = i0[t] XOR o0[t-1])" "0,1,1,0,1,1"
run_test "and" "05_and_gate.tau" "AND Gate (o0[t] = i0[t] & i1[t])" "0,0,0,1"

echo "============================================"
echo "Phase 1 Complete"
echo "============================================"

