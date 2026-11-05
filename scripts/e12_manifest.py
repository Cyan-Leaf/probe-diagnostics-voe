"""Build results/E12_token_budget/manifest.json from what is on disk.

Kept separate from the analysis because the counts and hashes must come from the filesystem
at build time, not be typed in — that is the same rule every other task's manifest follows.
"""

import glob
import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E12_token_budget"
CACHE = ROOT / "cache"

# (tag, model, size, frames, tokens). BASE is E8's own cache dir, reused rather than
# re-extracted so the cross-check in E12_PREREG.md §7 stays meaningful.
LADDERS = {
    "ltx_temporal": [
        ("BASE", 512, 49, 1792), ("S512F97", 512, 97, 3328), ("S512F193", 512, 193, 6400),
        ("S512F289", 512, 289, 9472), ("S512F337", 512, 337, 11008), ("S512F393", 512, 393, 12800),
    ],
    "cog_spatial": [
        ("BASE", 512, 49, 13312), ("S384F49", 384, 49, 7488),
        ("S256F49", 256, 49, 3328), ("S192F49", 192, 49, 1872),
    ],
}


def sha(p: str) -> str:
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def dir_for(model: str, tag: str, weights: str, variant: str) -> Path:
    prefix = "ltx" if model == "ltx" else "cogvideox"
    suffix = "" if variant == "plain" else f"_{variant}"
    t = "" if tag == "BASE" else f"_{tag}"
    return CACHE / f"{prefix}_feats_{weights}_intphys2{suffix}{t}"


def ver(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=120).stdout.strip()
    except Exception:
        return "unknown"


def main():
    counts = {}
    for model, ladder in (("ltx", "ltx_temporal"), ("cogvideox2b", "cog_spatial")):
        for tag, _s, _f, _t in LADDERS[ladder]:
            for w in ("real", "random"):
                for v in ("plain", "cj"):
                    d = dir_for(model, tag, w, v)
                    if d.exists():
                        counts[d.name] = len(list(d.glob("*.pt")))

    manifest = {
        "task": "E12: is LTX's weak real AUC caused by its token budget?",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": {
            "ltx": {"repo": "Lightricks/LTX-Video",
                    "revision": "8984fa25007f376c1a299016d0957a37a2f797bb",
                    "layers": 28, "vae": "32x spatial / 8x temporal, 128 ch",
                    "patch": 1, "n_params_b": 2.0},
            "cogvideox2b": {"repo": "zai-org/CogVideoX-2b",
                            "revision": "1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01",
                            "layers": 30, "vae": "8x spatial / 4x temporal, 16 ch",
                            "patch": 2, "n_params_b": 2.0},
        },
        "ladders": {
            k: [{"tag": t, "size": s, "frames": f, "tokens": tok} for t, s, f, tok in v]
            for k, v in LADDERS.items()},
        "geometry_note": (
            "LTX ladder varies frames at fixed size, so BOTH the token count and the feature "
            "dimension (F_lat*D) change — reported under two calibers, flat and timeavg, and "
            "only the dimension-controlled one can attribute a change to the budget "
            "(E12_PREREG.md §4.4). CogVideoX ladder varies size at fixed frames, so the "
            "feature dimension is constant and tokens are the only thing moving — but spatial "
            "detail changes with it (§4.2)."),
        "deviations": [
            {"what": "1024-size LTX rung is upsampled, not native",
             "why": "IntPhys2 is natively 512; 1024 gives 4 tokens per real pixel",
             "consequence": "used only for the Q2 crossover at ~13300 tokens, never for the "
                            "Q1 slope (E12_PREREG.md §4.3)"},
            {"what": "CogVideoX S=192 rung is downsampled",
             "why": "deliberate, to reach ~1800 tokens for the Q2 crossover",
             "consequence": "excluded from any slope fit; it measures lost detail, not budget "
                            "(§4.2)"},
            {"what": "LikePhys not covered",
             "why": "LikePhys clips are 56-61 frames, so LTX's 8x temporal compression caps "
                    "F_lat at 8 and no temporal ladder exists",
             "consequence": "E8's LikePhys LTX cell stands unchanged"}],
        "probe": {"folds": 5, "inner_folds": 3, "C_grid": [1e-3, 1e-2, 1e-1, 1.0, 10.0],
                  "grouping": "intphys2 scene_index", "feature_seeds": [0, 1, 2],
                  "threads": 4,
                  "threads_note": "must be 4; the probe is not bit-reproducible across BLAS "
                                  "thread counts (E16 found this). E12's BASE rungs reproduce "
                                  "E8 exactly at 4 threads, confirming the setting."},
        "bootstrap": {"n_boot": 1000, "resamples": "scenes with replacement, per fold",
                      "limitation": "resamples evaluation units on frozen out-of-fold "
                                    "predictions; it does NOT include refit noise, so the true "
                                    "CI is wider and the fail_token verdict is conservative"},
        "extraction_counts": counts,
        "env": {"python": ver("python3 -V"),
                "torch": ver('python3 -c "import torch;print(torch.__version__)" 2>/dev/null'),
                "diffusers": ver('python3 -c "import diffusers;print(diffusers.__version__)" '
                                 '2>/dev/null'),
                "gpu": "4x NVIDIA B200 (183 GB each)"},
        "scripts": {p: sha(p) for p in sorted(glob.glob("scripts/e12_*.py"))},
        "deliverables": ["E12_PREREG.md", "E12_cells_ltx.json", "E12_cells_cogvideox2b.json",
                         "E12_verdict.json", "E12_REPORT.md", "manifest.json"],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(counts, indent=1))
    print("wrote", OUT / "manifest.json")


if __name__ == "__main__":
    main()
