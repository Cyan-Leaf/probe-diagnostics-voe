#!/usr/bin/env bash
# E13 queue: V-JEPA 2 clean features, then the CogVideoX-2b t=0 reference cells.
#
# Runs concurrently with the E12 queue on the same cards. Safe because E12's LTX runs peak at
# ~28 GB of 183 GB and V-JEPA 2's ViT-L peak is 1.4 GB, and the cache dirs are disjoint.
#
# INVARIANT (this file broke it once, killed and rewritten 13:1x): one condition per GPU.
# The output dir is `vjepa_feats_<weights>_<dataset>[_cj]`, so two GPUs running the same
# (weights, variant) both write the same filenames — `if dst.exists(): continue` does not
# protect against two processes reaching the same clip at the same time, and a half-written
# .pt from one process would be skipped as "already done" by the other. Four conditions,
# four cards, no sharding. The four dirs are pairwise distinct:
#   real/plain, real/cj, random/plain, random/cj
#
# E13 is IntPhys2 only (E13_PREREG.md §8): LikePhys waits on the IntPhys2 verdicts.
set -u
cd "$(dirname "$0")/.." || exit 1
export HF_HOME=$HF_HOME
mkdir -p logs/gpu/e13
Q=logs/gpu/e13/queue.log

run() {  # run <gpu> <label> <script> <args...>
  local gpu=$1 label=$2 script=$3; shift 3
  echo "[$(date +%H:%M:%S)] GPU$gpu start $label" >> "$Q"
  CUDA_VISIBLE_DEVICES=$gpu python3 "$script" "$@" >> "logs/gpu/e13/g${gpu}_${label}.log" 2>&1
  echo "[$(date +%H:%M:%S)] GPU$gpu done  $label rc=$?" >> "$Q"
}

# CogVideoX-2b at t=0 on CLEAN latents: the only cell that makes the V-JEPA D1 comparable to
# anything in E8 (E13_PREREG.md §3.1). Both variants for one weight pair so the two runs
# that share a dir (T0 tag) stay on the same GPU.
cog_t0() {  # cog_t0 <gpu> <weights>
  local g=$1 w=$2
  for v in plain cj; do
    run "$g" "cog_t0_${w}_${v}" scripts/e12_extract.py --model cogvideox2b \
        --size 512 --frames 49 --dataset intphys2 --weights "$w" --variant "$v" \
        --t-grid 0 --tag T0
  done
}

( run 0 "vjepa_real_plain"   scripts/e13_extract.py --dataset intphys2 --weights real   --variant plain
  cog_t0 0 real ) &

( run 1 "vjepa_real_cj"      scripts/e13_extract.py --dataset intphys2 --weights real   --variant cj
  cog_t0 1 real ) &

( run 2 "vjepa_random_plain" scripts/e13_extract.py --dataset intphys2 --weights random --variant plain
  cog_t0 2 random ) &

( run 3 "vjepa_random_cj"    scripts/e13_extract.py --dataset intphys2 --weights random --variant cj
  cog_t0 3 random ) &

wait
echo "[$(date +%H:%M:%S)] queue complete" >> "$Q"
