#!/bin/bash
# Test all Tau Q-Agent specs

echo "========================================"
echo "Tau Q-Agent Test Suite"
echo "========================================"
echo ""

run_spec() {
    local name=$1
    local spec=$2
    echo "--- $name ---"
    cat "$spec" | docker run --rm -i tau:latest 2>&1 | grep -E "^(Execution|:=)" | head -8
    echo ""
}

echo "PHASE 1: Foundation"
echo "==================="
run_spec "Echo" "phase1/01_echo.tau"
run_spec "Negation" "phase1/02_negation.tau"
run_spec "Toggle" "phase1/04_toggle.tau"

echo "PHASE 2: Complex Logic"  
echo "======================"
run_spec "OR Gate" "phase2/06_or_gate.tau"
run_spec "Edge Detect" "phase2/07_edge_detect.tau"

echo "PHASE 3: Q-Agents"
echo "================="
run_spec "Deflationary Agent" "phase3/12_deflationary_agent.tau"
run_spec "Intelligent Q-Agent" "phase3/14_intelligent_q_agent.tau"

echo "========================================"
echo "All tests complete!"
echo "========================================"

