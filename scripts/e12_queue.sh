#!/usr/bin/env bash
# E12 extraction queue: 4 GPUs, greedy load balance by estimated cost.
#
# Every job is one (geometry, weights, variant) full extraction. Sharding is *not* used
# across GPUs here: a geometry's clips are independent, so running different geometries in
# parallel gives better utilisation than 4 shards of one job (the biggest job is ~30 min).
#
# Safe to re-run: records skip when the file exists, so an interrupted queue resumes.
set -u
cd "$(dirname "$0")/.." || exit 1
export HF_HOME=$HF_HOME
mkdir -p logs/gpu/e12

run() {  # run <gpu> <label> <args...>
  local gpu=$1 label=$2; shift 2
  echo "[$(date +%H:%M:%S)] GPU$gpu start $label" >> logs/gpu/e12/queue.log
  CUDA_VISIBLE_DEVICES=$gpu python3 scripts/e12_extract.py "$@" \
      >> "logs/gpu/e12/g${gpu}_${label}.log" 2>&1
  echo "[$(date +%H:%M:%S)] GPU$gpu done  $label rc=$?" >> logs/gpu/e12/queue.log
}

# ---- queue definitions (heaviest first so the tail does not serialise) -----------------
# LTX temporal ladder, size 512 fixed  (F -> F_lat -> tokens: 6->... see E12_PREREG §2.1)
ltx() { # ltx <gpu> <frames>
  local g=$1 f=$2
  for w in real random; do
    for v in plain cj; do
      run "$g" "ltx_S512F${f}_${w}_${v}" --model ltx --size 512 --frames "$f" \
          --dataset intphys2 --weights "$w" --variant "$v"
    done
  done
}
# CogVideoX-2b spatial ladder, frames 49 fixed
cog() { # cog <gpu> <size>
  local g=$1 s=$2
  for w in real random; do
    for v in plain cj; do
      run "$g" "cog_S${s}F49_${w}_${v}" --model cogvideox2b --size "$s" --frames 49 \
          --dataset intphys2 --weights "$w" --variant "$v"
    done
  done
}

# GPU0: heaviest LTX rungs
( ltx 0 393; ltx 0 193 ) &
# GPU1: second heaviest + the 1024 crossover point
( ltx 1 337; ltx 1 97 ) &
# GPU2: mid LTX rung + CogVideoX ladder (cheap)
( ltx 2 289; cog 2 384; cog 2 192 ) &
# GPU3: LTX 1024/97 crossover + remaining CogVideoX
( run 3 "ltx_S1024F97_real_plain"  --model ltx --size 1024 --frames 97 --dataset intphys2 --weights real   --variant plain
  run 3 "ltx_S1024F97_real_cj"     --model ltx --size 1024 --frames 97 --dataset intphys2 --weights real   --variant cj
  run 3 "ltx_S1024F97_random_plain" --model ltx --size 1024 --frames 97 --dataset intphys2 --weights random --variant plain
  run 3 "ltx_S1024F97_random_cj"    --model ltx --size 1024 --frames 97 --dataset intphys2 --weights random --variant cj
  cog 3 256 ) &
wait
echo "[$(date +%H:%M:%S)] queue complete" >> logs/gpu/e12/queue.log
