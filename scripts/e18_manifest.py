"""Build results/E18_own_band/manifest.json from what is on disk.

Deliberately reports the timestep of every cache dir, because that is the one thing that makes
E18's inputs different from E17's and the easiest thing for a reader to confuse.
"""
import glob
import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E18_own_band"
CACHE = ROOT / "cache"

OWN_BAND = {"low": 300, "high": 900}


def sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def n_pt(d):
    return len(list((CACHE / d).glob("*.pt")))


def ver(c):
    try:
        return subprocess.run(c, shell=True, capture_output=True, text=True,
                              timeout=120).stdout.strip()
    except Exception:
        return "unknown"


def dirname(expert, weights, variant, t):
    suffix = "" if variant == "plain" else f"_{variant}"
    tpart = "" if t == 600 else f"_t{t}"
    return f"wan22a14b_{expert}_feats_{weights}_intphys2{suffix}{tpart}"


dirs = [dirname(e, w, v, t) for e, t in OWN_BAND.items()
        for w in ("real", "random") for v in ("plain", "cj")]

m = {
    "task": ("E18: each Wan2.2-T2V-A14B expert measured inside its own noise band -- does "
             "noise-band specialisation change D1?"),
    "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    "prereg": "results/E18_own_band/E18_PREREG.md",
    "model": {"repo": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
              "revision": "5be7df9619b54f4e2667b2755bc6a756675b5cd7",
              "experts": {"high": {"subfolder": "transformer", "params_b": 14.0, "layers": 40,
                                   "noise_band": "sigma >= 0.875"},
                          "low": {"subfolder": "transformer_2", "params_b": 14.0, "layers": 40,
                                  "noise_band": "sigma < 0.875"}},
              "boundary_ratio": 0.875,
              "boundary_source": "model_index.json, verified on disk"},
    "design": {
        "cells": {e: {"t": t, "sigma": t / 1000.0,
                      "in_own_band": (t / 1000.0 >= 0.875) if e == "high" else (t / 1000.0 < 0.875)}
                  for e, t in OWN_BAND.items()},
        "why_diagonal_only": ("E18_PREREG.md §2: the off-diagonal cells answer generalisation, "
                              "and one point per expert cannot describe a generalisation curve"),
    },
    "data": {"dataset": "IntPhys2 Main", "n_plain": 1012, "n_cj": 506},
    "extraction_counts": {d: n_pt(d) for d in dirs},
    "probe": {"folds": 5, "inner_folds": 3, "C_grid": [1e-3, 1e-2, 1e-1, 1.0, 10.0],
              "grouping": "intphys2 scene_index", "feature_seeds": [0, 1, 2], "threads": 4,
              "implementation": "imported from e12_analyse.probe_oof / auc_of and "
                                "e8_verdict.interpretable -- not re-implemented (E11 lesson)"},
    "timestep_discipline": {
        "t600": "E17's delivered cells, read from results/E17_moe_experts/E17_cells.json, "
                "never recomputed here",
        "t300_t900": "new; cache dirs carry a _t<t> suffix so E17's features are untouched",
    },
    "comparison": {"delta_own": "D1(low@t=300) - D1(high@t=900)",
                   "delta_cross": "D1(low@t=600) - D1(high@t=600), from E17",
                   "threshold_pp": 2.51,
                   "threshold_source": "E16 probe-partition sd median (IntPhys2)"},
    "env": {"python": ver("python3 -V"),
            "torch": ver('python3 -c "import torch;print(torch.__version__)" 2>/dev/null'),
            "diffusers": ver('python3 -c "import diffusers;print(diffusers.__version__)" 2>/dev/null'),
            "gpu": "4x NVIDIA B200 (183 GB each)"},
    "scripts": {p: sha(p) for p in sorted(glob.glob("scripts/e18_*.py"))
                + sorted(glob.glob("scripts/e18_*.sh"))},
    "deliverables": ["E18_PREREG.md", "E18_cells.json", "E18_verdict.json", "E18_REPORT.md",
                     "manifest.json"],
}

(OUT / "manifest.json").write_text(json.dumps(m, indent=1))
print(json.dumps(m["extraction_counts"], indent=1))
print("wrote", OUT / "manifest.json")
