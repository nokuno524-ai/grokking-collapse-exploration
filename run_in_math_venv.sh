#!/bin/bash
# Wrapper to run any command in the math-grokking .venv environment.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/.venv/bin/activate"
exec "$@"
