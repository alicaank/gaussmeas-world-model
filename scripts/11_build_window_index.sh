#!/usr/bin/env bash
# Window table + window index v4 of the 864 corpus (CPU, ~20 min). Needs core/data864 (scripts/10_prepare_data864.sh).
#   contract/data_contract/windows.jsonl   every candidate window: per-call motion, anchor coverage, bank size, true revisits
#   contract/data_contract/index_v4.json   5 windows per stream, motion-weighted towards the inference range (the index committed in git is the one used)
set -euo pipefail; source "$(dirname "$0")/env.sh"; cd "$REPO/contract"
python data_contract/build_windows.py
python data_contract/build_index.py
