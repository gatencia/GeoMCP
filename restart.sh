#!/bin/bash
# restart.sh

set -e

API_PORT=8000
MCP_PORT=8765

echo "🔪 Killing any running servers on ports $API_PORT and $MCP_PORT..."
for port in "$API_PORT" "$MCP_PORT"; do
  if PIDS=$(lsof -ti tcp:$port); then
    if [ -n "$PIDS" ]; then
      echo "Found process(es) on port $port (PID: $PIDS). Killing..."
      kill -9 $PIDS || true
    fi
  fi
done

echo "🧼 Waiting a bit..."
sleep 1

echo "🚀 Starting GeoMCP FastAPI backend on port $API_PORT..."
source .venv/bin/activate
uvicorn server:app --reload --host 127.0.0.1 --port $API_PORT --reload-exclude '.venv/*' &
UVICORN_PID=$!
echo "Uvicorn PID: $UVICORN_PID"

echo "🚀 Starting FastMCP server on port $MCP_PORT..."
exec fastmcp run ./mcp_geomcp.py --transport streamable-http --host 127.0.0.1 --port $MCP_PORT