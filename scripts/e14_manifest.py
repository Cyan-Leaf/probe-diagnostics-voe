"""Build results/E14_hunyuan/manifest.json from what is on disk."""
import glob
import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E14_hunyuan"
CACHE = ROOT / "cache"


def sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def n_pt(d):
    return len(list((CACHE / d).glob("*.pt")))


def ver(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=120).stdout.strip()
    except Exception:
        return "unknown"


dirs = ["hunyuan_feats_real_intphys2", "hunyuan_feats_real_intphys2_cj",
        "hunyuan_feats_random_intphys2", "hunyuan_feats_random_intphys2_cj"]
m = {
    "task": "E14: fifth model family (HunyuanVideo) under the E8 protocol",
    "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    "model": {"repo": "hunyuanvideo-community/HunyuanVideo",
              "revision": "e8c2aaa66fe3742a32c11a6766aecbf07c56e773",
              "transformer_params_b": 12.8,
              "n_double_blocks": 20, "n_single_blocks": 40, "n_blocks": 60,
              "hidden": 3072, "heads": 24, "head_dim": 128,
              "text_encoder": "LlamaModel 8B (4096) + CLIPTextModel (768 pooled)",
              "vae": "AutoencoderKLHunyuanVideo, 8x spatial / 4x temporal, 16 ch",
              "has_vae": True, "has_noising_step": True},
    "geometry": {"height": 512, "width": 512, "frames": 49,
                 "latent_shape": [1, 16, 13, 64, 64],
                 "latent_layout": "B,C,F,H,W (NOT frame-major)",
                 "patch": [2, 1], "tokens": 13312},
    "data": {"dataset": "IntPhys2 Main", "n_plain": 1012, "n_cj": 506,
             "manifest": "results/E7_intphys2/manifest.json"},
    "extraction_counts": {d: n_pt(d) for d in dirs},
    "probe": {"folds": 5, "inner_folds": 3, "C_grid": [1e-3, 1e-2, 1e-1, 1.0, 10.0],
              "grouping": "intphys2 scene_index", "feature_seeds": [0, 1, 2]},
    "floors": "seed-spread (PREREG.md D4 definition A), same as E8's diffusion backbones",
    "deviations": [
        {"what": "L16 and Lrel=32 are in different block stacks",
         "why": "20 double-stream blocks then 40 single-stream",
         "consequence": "the two cells are NOT a depth contrast; the report says so every "
                        "time it shows them together"},
        {"what": "random DiT falls below raw pixel",
         "why": "unexplained; one model, posited norm/qk_norm interaction not tested",
         "consequence": "the trained>random>pixel ordering does not reproduce here; "
                        "ordering_reproduces is false in E14_verdict.json"}],
    "env": {"python": ver("python3 -V"),
            "torch": ver('python3 -c "import torch;print(torch.__version__)" 2>/dev/null'),
            "diffusers": ver('python3 -c "import diffusers;print(diffusers.__version__)" 2>/dev/null'),
            "gpu": "4x NVIDIA B200 (183 GB each)"},
    "scripts": {p: sha(p) for p in sorted(glob.glob("scripts/e14_*.py"))},
    "peak_vram_gb": 40.4,
    "s_per_clip": 4.08,
}
(OUT / "manifest.json").write_text(json.dumps(m, indent=1))
print(json.dumps(m["extraction_counts"], indent=1))
print("wrote", OUT / "manifest.json")
