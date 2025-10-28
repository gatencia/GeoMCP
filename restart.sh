#!/bin/bash
# restart.sh

PORT=8000

echo "🔪 Killing any running Uvicorn servers on port $PORT..."
PID=$(lsof -ti tcp:$PORT)
if [ -n "$PID" ]; then
  echo "Found process on port $PORT (PID: $PID). Killing..."
  kill -9 $PID
else
  echo "No process on port $PORT."
fi

echo "🧼 Waiting a bit..."
sleep 1

echo "🚀 Starting new GeoMCP server..."
source .venv/bin/activate
uvicorn server:app --reload --port $PORT