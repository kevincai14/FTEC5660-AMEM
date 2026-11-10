#!/usr/bin/env bash
# The k sweep, run after the main arm so that the memory caches already exist.
# Both arms load cached memories (the log line "Loading cached memories for sample 0"),
# so they differ from the main arm only in retrieve_k.
#
# Budget caps are per arm and are set above the measured expectation but below the
# account balance, so a surprise stops the run instead of draining the account:
#   k=5  ~ $0.16 (5.4k prompt tokens/question)
#   k=20 ~ $0.65 (21k prompt tokens/question)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
[ -x "$PY" ] || PY=.venv/bin/python

$PY repro/run.py --samples 1 --retrieve-k 5  --tag sweep5  --budget-usd 0.50
$PY repro/run.py --samples 1 --retrieve-k 20 --tag sweep20 --budget-usd 1.00

echo "--- paired comparisons on the 199 shared questions of conv-26 ---"
mkdir -p runs/tables
$PY repro/score.py runs/main_k10_s2_deepseek-flash_seed42 \
                   runs/sweep5_k5_s1_deepseek-flash_seed42 \
                   runs/sweep20_k20_s1_deepseek-flash_seed42 \
                   --compare --only-sample 0 --markdown runs/tables/table-sweep.md \
                   --tokens-md runs/tables/table-sweep-tokens.md \
                   --out runs/tables/sweep-score.json
