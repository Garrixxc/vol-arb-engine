#!/bin/bash
# Start the Vol-Arb Engine Dashboard
# Use the local .venv for Python dependencies

echo "Starting Aether Vol-Arb Engine Dashboard..."
echo "Access at: http://localhost:8050"

# Kill any existing process on 8050 just in case
lsof -ti:8050 | xargs kill -9 2>/dev/null

.venv/bin/python3 app.py
