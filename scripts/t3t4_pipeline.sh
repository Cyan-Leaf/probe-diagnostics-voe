#!/usr/bin/env bash
# T3/T4 orchestration: wait for the real-backbone extraction, then run the random-backbone
# extraction (CALIPER lower bound), consolidation, probes and analyses.
# Usage: bash scripts/t3t4_pipeline.sh
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
export HF_HOME=$HF_HOME
export HF_HUB_OFFLINE=1
LOGDIR="$ROOT/results/T3_null_floor/logs"
mkdir -p "$LOGDIR"

echo "[pipeline] waiting for real-backbone extraction to finish ..."
while true; do
  done_count=$( { grep -l "shard .* done in" "$LOGDIR"/extract_real_shard*.log 2>/dev/null || true; } | wc -l)
  n_files=$( { ls "$ROOT/cache/feats_real" 2>/dev/null || true; } | wc -l)
  if [ "$done_count" -ge 3 ]; then
    echo "[pipeline] real extraction complete: $n_files clip files"
    break
  fi
  sleep 60
done

echo "[pipeline] launching random-init backbone extraction (CALIPER lower bound) ..."
for i in 0 1 2; do
  gpu=$((i+1))
  CUDA_VISIBLE_DEVICES=$gpu nohup uv run --no-project python scripts/t3_extract.py \
      --shard "$i" --num-shards 3 --backbone random \
      > "$LOGDIR/extract_random_shard$i.log" 2>&1 &
done

while true; do
  done_count=$( { grep -l "shard .* done in" "$LOGDIR"/extract_random_shard*.log 2>/dev/null || true; } | wc -l)
  if [ "$done_count" -ge 3 ]; then
    echo "[pipeline] random extraction complete: $(ls "$ROOT/cache/feats_random" | wc -l) clip files"
    break
  fi
  sleep 30
done

echo "[pipeline] E0 null floor (full data) ..."
uv run --no-project python scripts/t3_null_floor.py 2>&1 | tee "$LOGDIR/e0_full.log"

echo "[pipeline] consolidating features ..."
uv run --no-project python scripts/t4_caliper.py --stage consolidate --backbone real \
    2>&1 | tail -n 12
uv run --no-project python scripts/t4_caliper.py --stage consolidate --backbone random \
    2>&1 | tail -n 12

echo "[pipeline] E1 probes ..."
uv run --no-project python scripts/t4_caliper.py --stage probe --workers 24 \
    2>&1 | tee "$LOGDIR/e1_probe.log" | tail -n 5

echo "[pipeline] summaries and plots ..."
uv run --no-project python scripts/t4_summarise.py 2>&1 | tail -n 40
uv run --no-project python scripts/make_plots.py --which e0 e1

echo "[pipeline] DONE"
