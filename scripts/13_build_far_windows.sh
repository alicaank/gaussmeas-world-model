#!/usr/bin/env bash
# List the windows farther than 10 model units from the anchor (CPU, 1 min) -> contract/data_contract/far_windows_v4.json (excluded from stage C).
set -euo pipefail; source "$(dirname "$0")/env.sh"; python "$REPO/contract/data_contract/build_far_windows.py" "$@"
