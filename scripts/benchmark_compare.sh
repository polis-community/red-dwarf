#!/usr/bin/env bash
set -euo pipefail

BENCH_ROOT=".benchmarks"

# --- Gather autosave benchmark files ---
FILES=$(find "$BENCH_ROOT" -name "*.json" | sort -r || true)

if [[ -z "$FILES" ]]; then
    echo "No benchmark files found in $BENCH_ROOT."
    exit 1
fi

# --- Get recent main commits ---
# Last 30 commits on main (newest first)
MAIN_COMMITS=($(git log main -n 30 --pretty=format:%h))

BASELINE_FILE=""
BASELINE_ID=""

if [[ -z "${NO_BASELINE:-}" ]]; then
    echo "Searching for a benchmark baseline matching a recent main commit..."
    for C in "${MAIN_COMMITS[@]}"; do
        # Look for autosaved benchmark matching this commit
        MATCH=$(echo "$FILES" | grep "$C" || true)
        if [[ -n "$MATCH" ]]; then
            # pick the newest match (first line)
            BASELINE_FILE=$(echo "$MATCH" | head -n 1)
            BASELINE_ID=$(basename "$BASELINE_FILE" .json)
            echo "Found baseline: $BASELINE_FILE"
            break
        fi
    done
else
    echo "NO_BASELINE is set — skipping baseline search."
fi

if [[ -z "$BASELINE_FILE" ]]; then
    echo "⚠️ No autosave benchmark found for recent main commits."
    echo "Running benchmarks without comparison."
    pytest tests --benchmark-only --benchmark-autosave
    exit 0
fi

echo "Using baseline ID: $BASELINE_ID"

pytest tests --benchmark-only --benchmark-autosave --benchmark-group-by=func --benchmark-compare="$BASELINE_ID"
