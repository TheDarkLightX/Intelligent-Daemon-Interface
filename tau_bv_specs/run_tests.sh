#!/bin/bash
# Tau Language Experiment Test Runner
# Tests specifications using Docker

TAU_IMAGE="tau:latest"
TAU_CMD="docker run --rm -i --entrypoint /tau-lang/build-Release/tau $TAU_IMAGE"

echo "============================================"
echo "Tau Language Specification Tests"
echo "============================================"
echo ""

# Test 1: Echo
echo "Test 1: Echo (output = input)"
echo "Input:  0, 1, 0"
echo "Expect: 0, 1, 0"
echo "-----------"
echo -e "sbf i1 = console\nsbf o1 = console\nr o1[t] = i1[t]\n0\n1\n0\nq" | $TAU_CMD 2>&1 | grep "o1\[" | head -3
echo ""

# Test 2: Negation
echo "Test 2: Negation (output = NOT input)"
echo "Input:  0, 1, 0"
echo "Expect: 1, 0, 1"
echo "-----------"
echo -e "sbf i1 = console\nsbf o1 = console\nr o1[t] = i1[t]'\n0\n1\n0\nq" | $TAU_CMD 2>&1 | grep "o1\[" | head -3
echo ""

# Test 3: Toggle
echo "Test 3: Toggle (toggle on input=1)"
echo "Input:  1, 0, 1, 1"
echo "Starting: 0"
echo "Expect: 0->1, 1->1, 1->0, 0->1"
echo "-----------"
echo -e "sbf i1 = console\nsbf o1 = console\nr o1[t] = (i1[t] & o1[t-1]') | (i1[t]' & o1[t-1])\n1\n0\n1\n1\nq" | $TAU_CMD 2>&1 | grep "o1\[" | head -5
echo ""

# Test 4: AND Gate
echo "Test 4: AND Gate (o1 = i1 AND i2)"
echo "Input: (0,0), (0,1), (1,1), (1,0)"
echo "Expect: 0, 0, 1, 0"
echo "-----------"
echo -e "sbf i1 = console\nsbf i2 = console\nsbf o1 = console\nr o1[t] = i1[t] & i2[t]\n0\n0\n0\n1\n1\n1\n1\n0\nq" | $TAU_CMD 2>&1 | grep "o1\[" | head -4
echo ""

# Test 5: XOR Gate
echo "Test 5: XOR Gate (o1 = i1 XOR i2)"
echo "Input: (0,0), (0,1), (1,0), (1,1)"
echo "Expect: 0, 1, 1, 0"
echo "-----------"
echo -e "sbf i1 = console\nsbf i2 = console\nsbf o1 = console\nr o1[t] = i1[t] + i2[t]\n0\n0\n0\n1\n1\n0\n1\n1\nq" | $TAU_CMD 2>&1 | grep "o1\[" | head -4
echo ""

# Test 6: Memory/Delay (output = previous input)
echo "Test 6: Memory/Delay (o1 = i1[t-1])"
echo "Input: 1, 0, 1, 0"
echo "Expect: ?, 1, 0, 1 (first is unspecified)"
echo "-----------"
echo -e "sbf i1 = console\nsbf o1 = console\nr o1[t] = i1[t-1]\n1\n0\n1\n0\nq" | $TAU_CMD 2>&1 | grep "o1\[" | head -4
echo ""

echo "============================================"
echo "All tests completed!"
echo "============================================"

