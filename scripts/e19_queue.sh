#!/usr/bin/env bash
# E19 queue: MiniMax-H3 (33B), four GPU processes, one task each.
#
# Why one process per GPU and not two: the smoke test measured 119.8 GB peak on the real
# weights and 133.4 GB on the random-init control, against a 183 GB card. Two processes would
# OOM. So the four (weights x variant) combinations are exactly four GPUs, and there is no
# second wave.
#
# Uses `.venv_h3`, NOT the system python: H3 needs diffusers 0.42.1 + transformers >= 4.60 +
# huggingface_hub >= 1.32, while the repo's other scripts need diffusers 0.36.0 +
# transformers 4.57.0 + hub < 1.0. The two sets are mutually exclusive, which is why E8-E18
# must keep running on `python3` and only E19 uses this interpreter.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv_h3/bin/python
export HF_HOME=$HF_HOME
# The transformers/diffusers HF stack reaches huggingface.co even for a local-only load; the
# proxy keeps that from hanging for a minute per component.
export http_proxy=${http_proxy:-http://<http-proxy>}
export https_proxy=${https_proxy:-http://<http-proxy>}

Q=logs/gpu/e19/queue.log
mkdir -p logs/gpu/e19

run() { local gpu=$1 label=$2; shift 2
  echo "[$(date +%H:%M:%S)] GPU$gpu start $label" >> "$Q"
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/e19_extract.py --dataset intphys2 "$@" \
      >> "logs/gpu/e19/g${gpu}_${label}.log" 2>&1
  local rc=$?
  echo "[$(date +%H:%M:%S)] GPU$gpu done  $label rc=$rc" >> "$Q"
  return $rc; }

run 0 real_plain    --weights real   --variant plain &
run 1 real_cj       --weights real   --variant cj &
run 2 random_plain  --weights random --variant plain &
run 3 random_cj     --weights random --variant cj &

wait
echo "[$(date +%H:%M:%S)] queue complete" >> "$Q"
echo "queue complete; next: $PY scripts/e19_analyse.py --threads 4"
