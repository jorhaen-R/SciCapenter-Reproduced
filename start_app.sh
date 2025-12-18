#!/usr/bin/env bash
set -e

if ! docker info >/dev/null 2>&1; then
  echo "Please start Docker Desktop (or Docker daemon) first!"
  exit 1
fi

echo "Starting SciCapenter containers..."
docker compose up -d

echo "Waiting for server to initialize..."
sleep 5

# Open browser
if [[ "$(uname)" == "Darwin" ]]; then
  open "http://localhost:8000"
else
  xdg-open "http://localhost:8000" >/dev/null 2>&1 || echo "Open http://localhost:8000 in your browser"
fi

echo "SciCapenter is running! Close this terminal to keep it running in background, or run 'docker compose down' to stop."
