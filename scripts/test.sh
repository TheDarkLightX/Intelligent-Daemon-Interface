#!/bin/bash
# Test the V28 Deflationary Agent with different scenarios

echo "======================================="
echo "Deflationary Agent V28 - Test Suite"
echo "======================================="

# Test directory
TEST_DIR="../test_scenarios"
mkdir -p $TEST_DIR

# Function to create test scenario
create_test() {
    local name=$1
    local desc=$2
    local price=$3
    local volume=$4
    local trend=$5
    
    echo ""
    echo "Test: $name - $desc"
    echo "------------------------"
    
    mkdir -p $TEST_DIR/$name/inputs
    mkdir -p $TEST_DIR/$name/outputs
    
    # Create input files
    echo "$price" > $TEST_DIR/$name/inputs/price.in
    echo "$volume" > $TEST_DIR/$name/inputs/volume.in
    echo "$trend" > $TEST_DIR/$name/inputs/trend.in
    
    # Copy specification
    cp ../specification/deflationary_agent_v28.tau $TEST_DIR/$name/
    
    # Modify paths in spec
    sed -i.bak 's|final_inputs|inputs|g' $TEST_DIR/$name/deflationary_agent_v28.tau
    sed -i.bak 's|final_outputs|outputs|g' $TEST_DIR/$name/deflationary_agent_v28.tau
    rm $TEST_DIR/$name/*.bak
    
    # Run test
    cd $TEST_DIR/$name
    docker run --rm \
        -v $(pwd)/inputs:/workdir/inputs \
        -v $(pwd)/outputs:/workdir/outputs \
        -v $(pwd):/workdir \
        -w /workdir \
        --entrypoint /tau-lang/build-Release/tau \
        tau-local deflationary_agent_v28.tau > /dev/null 2>&1
    
    # Analyze results
    if [ -f "outputs/profit.out" ]; then
        profits=$(grep -c "1" outputs/profit.out)
        burns=$(tail -1 outputs/burned.out)
        echo "✓ Completed: $profits profitable trades, $burns burns"
    else
        echo "✗ Test failed"
    fi
    
    cd - > /dev/null
}

# Test 1: Bull Market
create_test "bull_market" "Continuous opportunities" \
"0
0
0
0
0" \
"1
1
1
1
1" \
"1
1
1
1
1"

# Test 2: Bear Market  
create_test "bear_market" "No opportunities" \
"1
1
1
1
1" \
"0
0
0
0
0" \
"0
0
0
0
0"

# Test 3: Volatile Market
create_test "volatile" "Rapid changes" \
"0
1
0
1
0" \
"1
0
1
0
1" \
"1
0
1
0
1"

# Test 4: Perfect Conditions
create_test "perfect" "Ideal trading" \
"0
0
1
1
0" \
"1
1
0
0
1" \
"1
1
1
1
1"

echo ""
echo "======================================="
echo "Test suite completed!"
echo "======================================="