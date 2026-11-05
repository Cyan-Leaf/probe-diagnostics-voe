"""Build results/E13_non_generative/manifest.json from what is actually on disk.

Written as a script rather than by hand so the counts and the script hashes are read from the
filesystem at build time, not typed in.
"""

import glob
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "results" / "E13_non_generative"
CACHE = Path(__file__).resolve().parents[1] / "cache"


def sha(p: str) -> str:
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def n_pt(dirname: str) -> int:
    return len(list((CACHE / dirname).glob("*.pt")))


def ver(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=120).stdout.strip()
    except Exception:
        return "unknown"


def main():
    scripts = sorted(glob.glob("scripts/e13_*.py")) + [
        "scripts/e12_analyse.py", "scripts/e12_extract.py"]
    dirs = ["vjepa_feats_real_intphys2", "vjepa_feats_real_intphys2_cj",
            "vjepa_feats_random_intphys2", "vjepa_feats_random_intphys2_cj",
            "cogvideox_feats_real_intphys2_T0", "cogvideox_feats_real_intphys2_cj_T0",
            "cogvideox_feats_random_intphys2_T0", "cogvideox_feats_random_intphys2_cj_T0"]

    manifest = {
        "task": "E13: does the D1/D2 pattern survive on a NON-generative video encoder?",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": {"repo": "facebook/vjepa2-vitl-fpc64-256",
                  "revision": "b3c1679b7c34d3255ef3547f27c7b226aefab26f",
                  "params_b": 0.326, "layers": 24, "hidden": 1024, "patch": 16,
                  "tubelet": 2,
                  "training_objective": "joint-embedding prediction (JEPA), not denoising",
                  "has_vae": False, "has_noising_step": False},
        "geometry": {"height": 256, "width": 256, "frames": 64,
                     "resize_shortest_edge": 292, "tokens": 8192,
                     "why": "position encoding is fixed at 64 frames x 256; feeding anything "
                            "else either errors or silently interpolates (cf. the "
                            "CogVideoX-5b RoPE bug, E8 §12)"},
        "data": {"dataset": "IntPhys2 Main", "n_plain": 1012, "n_cj": 506,
                 "manifest": "results/E7_intphys2/manifest.json"},
        "extraction_counts": {d: n_pt(d) for d in dirs},
        "probe": {"folds": 5, "inner_folds": 3,
                  "C_grid": [1e-3, 1e-2, 1e-1, 1.0, 10.0],
                  "grouping": "intphys2 scene_index",
                  "seed": "feature seed fixed at 0 (deterministic forward)"},
        "floors": {"vjepa": "permutation within scenes, 20 draws (PREREG.md D4 definition B)",
                   "why_not_seed_spread": "a ViT forward has no noise draw, so the E8 "
                                          "seed-spread floor does not exist",
                   "diffusion_cells": "seed-spread, definition A, as in E8"},
        "deviations": [
            {"what": "no timestep", "why": "model has no noising schedule",
             "consequence": "D1 is defined on clean video and is NOT numerically comparable "
                            "to E8's noised D1; only verdict directions are compared"},
            {"what": "no VAE latent bound", "why": "model has no VAE",
             "consequence": "three lower bounds instead of four; recorded as null, not "
                            "substituted"},
            {"what": "256x256 instead of 512", "why": "position encoding is fixed",
             "consequence": "smaller pixel bandwidth for the physical evidence than E8; "
                            "stated in Limitations"},
            {"what": "floor definition differs", "why": "no feature seeds exist",
             "consequence": "E13 floors are not comparable to E8 floors"}],
        "env": {"python": ver("python3 -V"),
                "torch": ver('python3 -c "import torch;print(torch.__version__)" 2>/dev/null'),
                "transformers": ver('python3 -c "import transformers;'
                                    'print(transformers.__version__)" 2>/dev/null'),
                "gpu": "4x NVIDIA B200 (183 GB each)"},
        "scripts": {p: sha(p) for p in scripts},
        "deliverables": ["E13_PREREG.md", "E13_cells.json", "E13_verdict.json",
                         "E13_REPORT.md", "manifest.json"],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest["extraction_counts"], indent=1))
    print("wrote", OUT / "manifest.json")


if __name__ == "__main__":
    main()
