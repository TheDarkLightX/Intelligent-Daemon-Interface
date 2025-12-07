#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SEED="${SEED:-42}"
EPISODES="${EPISODES:-100}"
STEPS="${STEPS:-50}"

echo "Running intelligence_benchmark.py (seed=$SEED, episodes=$EPISODES, steps=$STEPS)"
cd "$SCRIPT_DIR"

SEED=$SEED EPISODES=$EPISODES STEPS=$STEPS python3 intelligence_benchmark.py


