#!/usr/bin/env sh
set -eu

script_dir=$(CDPATH= cd "$(dirname "$0")" && pwd)
repo_root=$(CDPATH= cd "$script_dir" && pwd)
port="${1:-8000}"

if ! command -v doxygen >/dev/null 2>&1; then
    printf '%s\n' "error: doxygen was not found in PATH" >&2
    exit 1
fi

python_cmd="${PYTHON:-python3}"
if ! command -v "$python_cmd" >/dev/null 2>&1; then
    if command -v python >/dev/null 2>&1; then
        python_cmd=python
    else
        printf '%s\n' "error: python3 was not found in PATH" >&2
        exit 1
    fi
fi

cd "$repo_root"

printf '%s\n' "Generating Doxygen documentation..."
doxygen docs/doxygen/Doxyfile.in

printf '\n%s\n' "Serving docs/doxygen/html at http://localhost:$port/"
printf '%s\n' "Press Ctrl-C to stop."
exec "$python_cmd" -m http.server "$port" --directory docs/doxygen/html
