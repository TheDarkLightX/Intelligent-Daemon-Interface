#!/bin/bash
# Run the V28 Deflationary Agent

echo "==================================="
echo "Deflationary Agent V28 - Execution"
echo "==================================="

# Check if we're in the right directory
if [ ! -f "specification/deflationary_agent_v28.tau" ]; then
    echo "Error: Must run from deflationary_agent_v28 directory"
    exit 1
fi

# Run the Tau specification
echo "Starting Tau execution..."
cd specification
docker run --rm \
    -v $(pwd)/../inputs:/workdir/final_inputs \
    -v $(pwd)/../outputs:/workdir/final_outputs \
    -v $(pwd):/workdir \
    -w /workdir \
    --entrypoint /tau-lang/build-Release/tau \
    tau-local deflationary_agent_v28.tau

# Check if execution was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "Execution completed successfully!"
    echo "Outputs written to outputs/ directory"
    
    # Run verification
    echo ""
    echo "Running verification..."
    cd ../verification
    python3 verify_v28.py
else
    echo "Execution failed!"
    exit 1
fi