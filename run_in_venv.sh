#!/bin/bash
# Wrapper to run any command in the .venv_real environment.
# Sandboxed-tool-friendly: executes the venv python via the activate script
# so the harness sees the wrapper as the entry point.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/.venv_real/bin/activate"
exec "$@"
