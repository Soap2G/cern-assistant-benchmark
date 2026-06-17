#!/bin/bash
# HTCondor job wrapper: runs one (system, family) slice of the benchmark on a
# worker node. Args: $1=system (lumi|accgpt)  $2=family (ops|cds)
#
# Self-contained: no AFS dependency, opencode from CVMFS, Python deps from a
# dedicated EOS venv, all scratch/state on the node-local _CONDOR_SCRATCH_DIR.
set -uo pipefail

SYSTEM="${1:?system arg required}"
FAMILY="${2:?family arg required}"
RUNID="bench_${SYSTEM}_${FAMILY}"

BENCH=/eos/user/g/gguerrie/benchmark
BENCH_VENV=/eos/user/g/gguerrie/bench-venv

# Node-local scratch for everything opencode/HOME/XDG writes (NOT EOS/AFS).
SCRATCH="${_CONDOR_SCRATCH_DIR:-${TMPDIR:-/tmp}}"
export HOME="$SCRATCH/home"
export XDG_DATA_HOME="$SCRATCH/xdg-data"
export XDG_CACHE_HOME="$SCRATCH/xdg-cache"
export XDG_CONFIG_HOME="$SCRATCH/xdg-config"
export LUMI_DATA_HOME="$SCRATCH/lumi-bench-data"   # consumed by adapters/lumi.py
mkdir -p "$HOME" "$XDG_DATA_HOME" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$LUMI_DATA_HOME"

# Lumi: opencode binary (CVMFS), config snapshot (EOS, AFS fallback), key (EOS).
export OPENCODE_BIN=/cvmfs/sw.escape.eu/lumi/latest/bin/opencode
export OPENCODE_CONFIG_DIR=/eos/user/g/gguerrie/lumi-assistant-config
[ -d "$OPENCODE_CONFIG_DIR" ] || export OPENCODE_CONFIG_DIR=/afs/cern.ch/user/g/gguerrie/lumi-assistant/config
export LUMI_CONFIG_DIR="$OPENCODE_CONFIG_DIR"
export LUMI_KEY_FILE=/eos/user/g/gguerrie/lumi_assistant/key.txt
export LITELLM_API_KEY="$(cat "$LUMI_KEY_FILE" 2>/dev/null)"
export LUMI_MODEL="litellm/gpt-4.1"
export LUMI_TIMEOUT=300

# accGPT (token read from a key file kept out of git)
export ACCGPT_API_URL=https://accgpt-ui.app.cern.ch
export ACCGPT_KEY_FILE=/eos/user/g/gguerrie/lumi_assistant/accgpt_key.txt
export ACCGPT_API_KEY="$(cat "$ACCGPT_KEY_FILE" 2>/dev/null)"

echo "=========================================================="
echo "[run_bench] $(date)"
echo "[run_bench] host=$(hostname)  system=$SYSTEM  family=$FAMILY  runid=$RUNID"
echo "[run_bench] scratch=$SCRATCH"
echo "[run_bench] OPENCODE_CONFIG_DIR=$OPENCODE_CONFIG_DIR (exists: $(test -d "$OPENCODE_CONFIG_DIR" && echo yes || echo NO))"
echo "[run_bench] opencode: $(test -x "$OPENCODE_BIN" && echo OK || echo MISSING)"
echo "[run_bench] venv python: $($BENCH_VENV/bin/python --version 2>&1)"
echo "[run_bench] key loaded: $([ -n "$LITELLM_API_KEY" ] && echo yes || echo NO)"
echo "=========================================================="

cd "$BENCH" || { echo "[run_bench] cannot cd to $BENCH"; exit 2; }

"$BENCH_VENV/bin/python" runner.py --system "$SYSTEM" --family "$FAMILY" --run-id "$RUNID"
rc=$?

echo "[run_bench] runner exit=$rc  $(date)"
# Leave a marker so the merge step knows this slice finished.
echo "$rc" > "$BENCH/results/${RUNID}.done"
exit $rc
