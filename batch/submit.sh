#!/bin/bash
# Submit the 4 benchmark jobs to HTCondor.
# Run this AFTER your Kerberos ticket + condor credmon are working
# (i.e. `condor_submit` no longer errors with "credmon did not process
# credentials"). A quick way to confirm: `condor_q` returns without error.
set -e
cd "$(dirname "$0")"

mkdir -p logs
chmod +x run_bench.sh

# Clear stale completion markers / old slice outputs from a previous run.
rm -f ../results/bench_lumi_ops*.jsonl ../results/bench_lumi_cds*.jsonl \
      ../results/bench_accgpt_ops*.jsonl ../results/bench_accgpt_cds*.jsonl \
      ../results/bench_*.done 2>/dev/null || true

echo "Submitting 4 jobs (lumi/accgpt x ops/cds)..."
condor_submit bench.sub

echo
echo "Monitor:    condor_q"
echo "Live logs:  tail -f logs/*.out"
echo "Results:    /eos/user/g/gguerrie/benchmark/results/bench_<system>_<family>.jsonl"
echo
echo "When condor_q is empty (4 .done markers in ../results), build the report:"
echo "    ./merge_report.sh"
