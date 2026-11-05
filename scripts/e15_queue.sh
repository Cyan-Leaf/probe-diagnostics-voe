#!/usr/bin/env bash
# E15 queue: LikePhys half of the cross-paradigm test (E15_PREREG.md).
# Four conditions, four cards, no sharding — same invariant as e13/e14 queues: one
# (model, weights, variant) per GPU so no two processes write the same filenames.
# No `cj` variant: LikePhys carries its own `appearance` group (E7/E8 use it for D2).
set -u
cd "$(dirname "$0")/.." || exit 1
export HF_HOME=$HF_HOME
Q=logs/gpu/e15/queue.log
run() { local gpu=$1 label=$2 script=$3; shift 3
  echo "[$(date +%H:%M:%S)] GPU$gpu start $label" >> "$Q"
  CUDA_VISIBLE_DEVICES=$gpu python3 "$script" "$@" >> "logs/gpu/e15/g${gpu}_${label}.log" 2>&1
  echo "[$(date +%H:%M:%S)] GPU$gpu done  $label rc=$?" >> "$Q"; }

( run 0 vjepa_likephys_real_plain   scripts/e13_extract.py --model vitl --dataset likephys --weights real   --variant plain ) &
( run 1 vjepa_likephys_random_plain scripts/e13_extract.py --model vitl --dataset likephys --weights random --variant plain ) &
( run 2 hunyuan_likephys_real_plain   scripts/e14_extract.py --dataset likephys --weights real   --variant plain ) &
( run 3 hunyuan_likephys_random_plain scripts/e14_extract.py --dataset likephys --weights random --variant plain ) &
wait
echo "[$(date +%H:%M:%S)] queue complete" >> "$Q"
