#!/bin/sh
set -eu
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$PROJECT_DIR"
if [ ! -x "$PROJECT_DIR/.venv/bin/python" ]; then
    printf '%s\n' 'Create the project environment first: python3 -m venv .venv'
    exit 1
fi
exec "$PROJECT_DIR/.venv/bin/python" -m streamlit run "$PROJECT_DIR/app.py" "$@"
