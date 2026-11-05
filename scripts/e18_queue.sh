#!/usr/bin/env bash
# E18 queue: each Wan2.2 expert measured INSIDE its own noise band.
#
# E17 ran both experts at t=600 (sigma 0.600 < boundary 0.875), so the low-noise expert was
# at home and the high-noise expert was out of range. That comparison cannot separate
# "specialises in its band" from "generalises to a band it never trained on". This queue
# supplies the missing half: low@t=300 (sigma 0.300) and high@t=900 (sigma 0.900), the two
# diagonal cells of E18_PREREG.md §2.
#
# Same invariant as e13/e14/e17: one (expert, weights, variant, t) per GPU, cache dir
# wan22a14b_<expert>_feats_<weights>_intphys2[_cj][_t<t>]. t=300/900 get the suffix, so
# E17's delivered t=600 features are untouched.
set -u
cd "$(dirname "$0")/.." || exit 1
export HF_HOME=$HF_HOME
Q=logs/gpu/e18/queue.log
mkdir -p logs/gpu/e18

run() { local gpu=$1 label=$2 t=$3; shift 3
  echo "[$(date +%H:%M:%S)] GPU$gpu start $label t=$t" >> "$Q"
  CUDA_VISIBLE_DEVICES=$gpu python3 scripts/e17_extract.py --t "$t" "$@" \
      >> "logs/gpu/e18/g${gpu}_${label}.log" 2>&1
  local rc=$?
  echo "[$(date +%H:%M:%S)] GPU$gpu done  $label rc=$rc" >> "$Q"
  return $rc; }

# Low-noise expert at its home band. Weights/variants split across GPU0/GPU1.
( run 0 low_real_plain  300 --expert low  --dataset intphys2 --weights real   --variant plain
  run 0 low_real_cj     300 --expert low  --dataset intphys2 --weights real   --variant cj ) &
( run 1 low_random_plain 300 --expert low --dataset intphys2 --weights random --variant plain
  run 1 low_random_cj   300 --expert low  --dataset intphys2 --weights random --variant cj ) &

# High-noise expert at its home band.
( run 2 high_real_plain  900 --expert high --dataset intphys2 --weights real   --variant plain
  run 2 high_real_cj     900 --expert high --dataset intphys2 --weights real   --variant cj ) &
( run 3 high_random_plain 900 --expert high --dataset intphys2 --weights random --variant plain
  run 3 high_random_cj   900 --expert high --dataset intphys2 --weights random --variant cj ) &

wait
echo "[$(date +%H:%M:%S)] queue complete" >> "$Q"
echo "queue complete; now run: python scripts/e18_analyse.py --threads 4 && python scripts/e18_verdict.py"
