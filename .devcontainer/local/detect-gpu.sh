#!/usr/bin/env bash
# Runs as initializeCommand before the devcontainer is built.
# Detects NVIDIA GPU availability and writes a project-root .env file so that
# docker-compose picks up the CUDA build arg automatically.
#
# The .env file is gitignored; this script is the source of truth.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="$PROJECT_ROOT/.env"

if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null 2>&1; then
    echo "GPU detected — enabling CUDA support (CUDA=true)."
    echo "CUDA=true" > "$ENV_FILE"
else
    echo "No GPU detected — using CPU-only mode (CUDA=false)."
    echo "CUDA=false" > "$ENV_FILE"
fi
