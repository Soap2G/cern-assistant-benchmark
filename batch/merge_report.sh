#!/bin/bash
# Merge the 4 per-slice result files into one combined run and build the report.
# Run this once all 4 jobs have finished (condor_q empty / 4 *.done markers).
set -e
BENCH=/eos/user/g/gguerrie/benchmark
VENV=/eos/user/g/gguerrie/bench-venv
cd "$BENCH/results"

slices="bench_lumi_ops bench_lumi_cds bench_accgpt_ops bench_accgpt_cds"

echo "Slice status:"
for s in $slices; do
  done=$( [ -f "${s}.done" ] && cat "${s}.done" || echo "MISSING" )
  rows=$( [ -f "${s}.jsonl" ] && wc -l < "${s}.jsonl" || echo 0 )
  echo "  $s : exit=$done, rows=$rows"
done

# Concatenate present slices (missing ones are simply skipped).
: > bench_all.jsonl
: > bench_all_scores.jsonl
for s in $slices; do
  [ -f "${s}.jsonl" ]        && cat "${s}.jsonl"        >> bench_all.jsonl
  [ -f "${s}_scores.jsonl" ] && cat "${s}_scores.jsonl" >> bench_all_scores.jsonl
done

echo
echo "Combined: $(wc -l < bench_all.jsonl) results, $(wc -l < bench_all_scores.jsonl) scores"
echo "Building report..."
"$VENV/bin/python" "$BENCH/report.py" "$BENCH/results/bench_all.jsonl"
echo
echo "Report artifacts written to $BENCH/results/ (bench_all_*.png, bench_all_summary.txt)"
