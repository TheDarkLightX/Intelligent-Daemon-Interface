#!/bin/bash
# Deflationary Agent - Complete execution trace
# i1 = buy_signal, i2 = sell_signal
# o1 = holding, o2 = burn

echo "============================================"
echo "Deflationary Agent Execution Trace"
echo "============================================"
echo ""
echo "Agent Spec:"
echo "  o1[t] = (i1[t] & o1[t-1]') | (o1[t-1] & i2[t]')"
echo "     -> holding = (buy AND NOT was_holding) OR (was_holding AND NOT sell)"
echo "  o2[t] = i2[t] & o1[t-1]"
echo "     -> burn = sell AND was_holding"
echo ""
echo "Scenario: Buy -> Hold -> Sell (BURN!) -> Buy again"
echo ""

# Clean trace with explicit steps
docker run --rm -i --entrypoint /tau-lang/build-Release/tau tau:latest 2>&1 <<'EOF'
sbf i1 = console
sbf i2 = console
sbf o1 = console
sbf o2 = console
r o1[t] = (i1[t] & o1[t-1]') | (o1[t-1] & i2[t]') && o2[t] = i2[t] & o1[t-1]
1
0
0
0
0
1
1
0
q
EOF

