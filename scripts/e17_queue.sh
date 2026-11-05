#!/usr/bin/env bash
# E17 queue: Wan2.2-T2V-A14B, two experts x two weights, one condition per GPU.
# Invariant (same as e13/e14): the cache dir is wan22a14b_<expert>_feats_<weights>_<dataset>[_cj],
# so one (expert, weights, variant) per GPU. No sharding.
# cj variants run after the plain ones on the same GPU to keep dirs disjoint.
set -u
cd "$(dirname "$0")/.." || exit 1
export HF_HOME=$HF_HOME
Q=logs/gpu/e17/queue.log
run() { local gpu=$1 label=$2; shift 2
  echo "[$(date +%H:%M:%S)] GPU$gpu start $label" >> "$Q"
  CUDA_VISIBLE_DEVICES=$gpu python3 scripts/e17_extract.py "$@" >> "logs/gpu/e17/g${gpu}_${label}.log" 2>&1
  echo "[$(date +%H:%M:%S)] GPU$gpu done  $label rc=$?" >> "$Q"; }

( run 0 low_real_plain   --expert low  --dataset intphys2 --weights real   --variant plain
  run 0 low_real_cj      --expert low  --dataset intphys2 --weights real   --variant cj ) &
( run 1 low_random_plain --expert low  --dataset intphys2 --weights random --variant plain
  run 1 low_random_cj    --expert low  --dataset intphys2 --weights random --variant cj ) &
( run 2 high_real_plain  --expert high --dataset intphys2 --weights real   --variant plain
  run 2 high_real_cj     --expert high --dataset intphys2 --weights real   --variant cj ) &
( run 3 high_random_plain --expert high --dataset intphys2 --weights random --variant plain
  run 3 high_random_cj    --expert high --dataset intphys2 --weights random --variant cj ) &
wait
echo "[$(date +%H:%M:%S)] queue complete" >> "$Q"
