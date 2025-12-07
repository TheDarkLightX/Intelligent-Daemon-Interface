#!/bin/bash
# Run benchmarks for all deflationary agent versions
# Copyright DarkLightX/Dana Edwards

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SPEC_DIR="$PROJECT_DIR/specification"

echo "=== Deflationary Agent Benchmark Suite ==="
echo "Project: $PROJECT_DIR"
echo "Specs: $SPEC_DIR"
echo ""

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 is required"
    exit 1
fi

# List specifications to benchmark
echo "Specifications found:"
for spec in "$SPEC_DIR"/agent4_testnet_v*.tau; do
    if [ -f "$spec" ]; then
        echo "  - $(basename "$spec")"
    fi
done
echo ""

# Run Python benchmark
echo "Running benchmark suite..."
python3 "$SCRIPT_DIR/benchmark_versions.py"

# Check for tau binary (optional - for native tau benchmarks)
TAU_BIN=""
if [ -f "$PROJECT_DIR/tau_daemon_alpha/bin/tau" ]; then
    TAU_BIN="$PROJECT_DIR/tau_daemon_alpha/bin/tau"
    echo ""
    echo "Tau binary found: $TAU_BIN"
    echo "Note: For native tau benchmarks, build tau-lang from source"
fi

# Summary
echo ""
echo "=== Benchmark Complete ==="
echo "Results saved to: $SCRIPT_DIR/benchmark_results.json"
echo ""
echo "To analyze results:"
echo "  cat $SCRIPT_DIR/benchmark_results.json | python3 -m json.tool"

