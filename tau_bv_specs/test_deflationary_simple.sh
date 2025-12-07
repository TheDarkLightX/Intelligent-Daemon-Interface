#!/bin/bash
# Simple Deflationary Agent Test - SBF version
# Tests position management and burn mechanism

TAU_IMAGE="tau:latest"
TAU_CMD="docker run --rm -i --entrypoint /tau-lang/build-Release/tau $TAU_IMAGE"

echo "============================================"
echo "Simple Deflationary Agent Test"
echo "============================================"
echo ""
echo "Agent behavior:"
echo "- Buys when: buy_signal=1 AND NOT holding"
echo "- Sells when: sell_signal=1 AND holding"
echo "- Burns on sell"
echo ""

INPUT="sbf buy_signal = console
sbf sell_signal = console
sbf holding = console
sbf burn = console
r holding[t] = (buy_signal[t] & holding[t-1]') | (holding[t-1] & sell_signal[t]') && burn[t] = sell_signal[t] & holding[t-1]
1
0
0
0
0
0
0
1
0
0
1
0
q"

echo "Scenario:"
echo "t=0: buy_signal=1, sell_signal=0 -> BUY, hold=1, burn=0"
echo "t=1: buy_signal=0, sell_signal=0 -> HOLD, hold=1, burn=0"
echo "t=2: buy_signal=0, sell_signal=1 -> SELL, hold=0, burn=1 (BURN!)"
echo "t=3: buy_signal=0, sell_signal=0 -> flat, hold=0, burn=0"
echo "t=4: buy_signal=1, sell_signal=0 -> BUY, hold=1, burn=0"
echo ""
echo "Running..."
echo ""

echo "$INPUT" | $TAU_CMD 2>&1 | grep -E "(holding|burn)\[" | head -12

echo ""
echo "============================================"

