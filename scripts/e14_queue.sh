#!/usr/bin/env bash
# E14 queue: HunyuanVideo, four conditions on four cards, no sharding.
#
# Same invariant as e13_queue.sh: the output dir is `hunyuan_feats_<weights>_<dataset>[_cj]`,
# so one (weights, variant) per GPU. Two GPUs on the same condition would race on the same
# filenames, and `if dst.exists(): continue` does not protect a half-written .pt.
#
# 3036 clip-forwards total at ~4 s each = ~3.4 GPU-hours, ~50 min wall on 4 cards.
set -u
cd "$(dirname "$0")/.." || exit 1
export HF_HOME=$HF_HOME
mkdir -p logs/gpu/e14
Q=logs/gpu/e14/queue.log

run() {  # run <gpu> <label> <args...>
  local gpu=$1 label=$2; shift 2
  echo "[$(date +%H:%M:%S)] GPU$gpu start $label" >> "$Q"
  CUDA_VISIBLE_DEVICES=$gpu python3 scripts/e14_extract.py "$@" \
      >> "logs/gpu/e14/g${gpu}_${label}.log" 2>&1
  echo "[$(date +%H:%M:%S)] GPU$gpu done  $label rc=$?" >> "$Q"
}

( run 0 hunyuan_real_plain   --dataset intphys2 --weights real   --variant plain ) &
( run 1 hunyuan_real_cj      --dataset intphys2 --weights real   --variant cj ) &
( run 2 hunyuan_random_plain --dataset intphys2 --weights random --variant plain ) &
( run 3 hunyuan_random_cj    --dataset intphys2 --weights random --variant cj ) &

wait
echo "[$(date +%H:%M:%S)] queue complete" >> "$Q"
