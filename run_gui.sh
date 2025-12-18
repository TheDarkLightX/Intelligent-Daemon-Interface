#!/bin/bash
# IDI GUI Launcher

# Ensure we are in the root directory
if [ ! -d "idi" ]; then
    echo "Error: Please run this script from the project root (Intelligent-Daemon-Interface)."
    exit 1
fi

export PYTHONPATH=$PYTHONPATH:$(pwd)

# Activate virtual environment
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Check backend dependencies
# (Optional: python -c "import fastapi" || pip install -r idi/gui/backend/requirements.txt)

echo "Starting Backend [Port 8000]..."
python3 -m uvicorn idi.gui.backend.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

echo "Starting Frontend [Port 3000]..."
cd idi/gui/frontend
npm run dev -- -p 3000 &
FRONTEND_PID=$!

echo "GUI launched! Open http://localhost:3000"
echo "Press Ctrl+C to stop."

# Wait for processes
wait $BACKEND_PID $FRONTEND_PID
