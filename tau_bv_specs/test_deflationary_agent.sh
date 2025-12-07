#!/bin/bash
# Deflationary Agent Test - SBF version
# This tests a simplified deflationary agent using simple Boolean functions
# Price signals: low_price (0) / high_price (1)
# Volume: low (0) / high (1)
# Trend: bearish (0) / bullish (1)

TAU_IMAGE="tau:latest"
TAU_CMD="docker run --rm -i --entrypoint /tau-lang/build-Release/tau $TAU_IMAGE"

echo "============================================"
echo "Deflationary Agent Test (SBF version)"
echo "============================================"
echo ""
echo "Testing trading agent that:"
echo "- Buys when: low_price=1 AND volume=1 AND trend=1 AND NOT holding"
echo "- Sells when: high_price=1 AND holding"
echo "- Burns tokens on profitable exits"
echo ""

# The deflationary agent spec
SPEC='
sbf low_price = console.
sbf high_price = console.
sbf volume = console.
sbf trend = console.
sbf holding = console.
sbf buy_price_stored = console.
sbf action = console.
sbf is_buy = console.
sbf should_burn = console.

# Buy signal: low price, high volume, bullish, not holding
is_buy[t] = low_price[t] & volume[t] & trend[t] & holding[t-1]'

# Action: buy or sell
action[t] = is_buy[t] | (high_price[t] & holding[t-1])

# Position state: enter on buy, exit on sell
holding[t] = (is_buy[t] & action[t]) | (holding[t-1] & (action[t]'"'"' | is_buy[t]))

# Buy price memory (simplified - 1 if entered at high, 0 if low)
buy_price_stored[t] = (is_buy[t] & action[t] & low_price[t]'"'"') | (action[t]'"'"' & buy_price_stored[t-1]) | (is_buy[t] & buy_price_stored[t-1])

# Should burn: selling (high_price and was holding) and profitable (bought low, sell high)
should_burn[t] = high_price[t] & holding[t-1] & buy_price_stored[t-1]'"'"'
'

echo "Test Scenario:"
echo "Step 0: Low price=1, High price=0, Volume=1, Trend=1 -> Should BUY"
echo "Step 1: Low price=0, High price=0, Volume=0, Trend=0 -> Hold"
echo "Step 2: Low price=0, High price=1, Volume=0, Trend=0 -> Should SELL (profitable, burn!)"
echo "Step 3: Low price=1, High price=0, Volume=1, Trend=1 -> Should BUY again"
echo ""

# Run the test
echo -e "sbf low_price = console
sbf high_price = console
sbf volume = console
sbf trend = console
sbf holding = console
sbf action = console
sbf is_buy = console
sbf should_burn = console
r is_buy[t] = low_price[t] & volume[t] & trend[t] & holding[t-1]' && action[t] = is_buy[t] | (high_price[t] & holding[t-1]) && holding[t] = (is_buy[t] & action[t]) | (holding[t-1] & action[t]') && should_burn[t] = high_price[t] & holding[t-1] & is_buy[t]'
1
0
1
1
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
1
1
q" | $TAU_CMD 2>&1 | grep -E "(is_buy|action|holding|should_burn)\[" | head -20

echo ""
echo "============================================"
echo "Test completed!"
echo "============================================"

