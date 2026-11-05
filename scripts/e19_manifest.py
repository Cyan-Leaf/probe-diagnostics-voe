"""Build results/E19_h3_scale/manifest.json from what is on disk.

E19's environment is the one thing that makes it unlike every other task in the repo: it runs
on `.venv_h3` (diffusers 0.42.1 / transformers 5.x / hub 2.x) while E8-E18 must run on the
system python (diffusers 0.36.0 / transformers 4.57 / hub 1.x). Those two sets are mutually
exclusive, so the manifest records BOTH interpreters and says which script ran under which.
A reader who reproduces E19 with `python3` will get an ImportError, not a wrong number -- but
they deserve to know before they try.
"""

import glob
import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E19_h3_scale"
CACHE = ROOT / "cache"
H3 = Path("$HF_HOME/minimax_h3")

DIRS = ["minimax_h3_feats_real_intphys2", "minimax_h3_feats_real_intphys2_cj",
        "minimax_h3_feats_random_intphys2", "minimax_h3_feats_random_intphys2_cj"]


def sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def n_pt(d):
    return len(list((CACHE / d).glob("*.pt")))


def ver(cmd, cwd=None):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=180,
                              cwd=cwd).stdout.strip().splitlines()[-1]
    except Exception:  # noqa: BLE001
        return "unknown"


def du_gb(p):
    try:
        out = subprocess.run(f"du -sb {p}", shell=True, capture_output=True, text=True,
                             timeout=300).stdout.split()[0]
        return round(int(out) / 2**30, 2)
    except Exception:  # noqa: BLE001
        return None


m = {
    "task": ("E19: MiniMax-H3 (33.1B) under the E8 protocol -- the size axis, plus a new text "
             "encoder (Qwen3-VL-32B) and a new VAE layout (24ch, f16t4d24)"),
    "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    "prereg": "results/E19_h3_scale/E19_PREREG.md",
    "model": {
        "repo": "MiniMaxAI/MiniMax-H3",
        "local_root": str(H3),
        "transformer": {"dir": "transformer", "format": "diffusers (sharded)",
                        "params_b": 33.1, "layers": 50, "hidden": 5376,
                        "heads": "56 x 128", "packed_sequence": True,
                        "note": "dense single-stream Omni Transformer; text/audio/video share one sequence"},
        "vae": {"dir": "vae", "format": "diffusers AutoencoderKLMiniMaxH3",
                "latent_channels": 24, "compression": "f16t4d24",
                "frames": "17*n+5 pixel frames -> 5*n+2 latent frames"},
        "text_encoder": {"dir": "text_encoder", "model": "Qwen3-VL-32B",
                         "note": "downloaded; NOT used for E19 -- the probe conditions on a "
                                 "single dummy text row, as all backbones do"},
        "raw_checkpoint_also_on_disk": {
            "dir": "FL2VA", "why": ("the original checkpoint. Its transformer keys are "
                                    "`blocks.*` / `final_layer.*` and diffusers cannot load it, "
                                    "so `transformer/` (diffusers format) is what E19 uses. "
                                    "Kept because the FL2VA video_vae wrapper is the fallback "
                                    "loader in e19_extract.py.")},
    },
    "design": {
        "reference_cell": {"layer": 16, "t": 600},
        "relative_depth_cell": {"layer": 27, "why": "16/30 of the stack, E8's convention, on 50 layers"},
        "frame_geometry": {"pixel_frames": 39, "groups": [17, 17, 5], "latent_frames": 12,
                           "why_not_49": ("49 is not encodable: the VAE accepts 17*n+5 frames "
                                          "only, and 49 would snap to 56 (17 latents), changing "
                                          "temporal resolution relative to E8. 39 gives 12 "
                                          "latents, matching the 13 the other backbones get.")},
        "source_frame_request": 49,
        "pixel_convention": ("ImageNet-normalised RGB over [0,1], NOT Wan's [-1,1]. The extractor "
                             "converts; `raw_pixel_32` still uses [-1,1] because the pixel arm is "
                             "recomputed in the analysis on H3's own 17/17/5 binning."),
        "vae_posterior": "sampled under a fixed generator (seed 42), then fp16-rounded - the released recipe",
    },
    "data": {"dataset": "IntPhys2 Main", "n_plain": 1012, "n_cj": 506},
    "extraction_counts": {d: n_pt(d) for d in DIRS},
    "sizes_gb": {"transformer": du_gb(H3 / "transformer"), "vae": du_gb(H3 / "vae"),
                 "text_encoder": du_gb(H3 / "text_encoder")},
    "probe": {"folds": 5, "inner_folds": 3, "C_grid": [1e-3, 1e-2, 1e-1, 1.0, 10.0],
              "grouping": "intphys2 scene_index", "feature_seeds": [0, 1, 2], "threads": 4,
              "implementation": "imported from e12_analyse.probe_oof / auc_of and "
                                "e8_verdict.interpretable -- never re-implemented (E11 lesson)"},
    "env": {
        "e19_interpreter": ".venv_h3/bin/python",
        "e19_versions": {
            "python": ver(".venv_h3/bin/python -V", cwd=str(ROOT)),
            "torch": ver('.venv_h3/bin/python -c "import torch;print(torch.__version__)"', cwd=str(ROOT)),
            "diffusers": ver('.venv_h3/bin/python -c "import diffusers;print(diffusers.__version__)"', cwd=str(ROOT)),
            "transformers": ver('.venv_h3/bin/python -c "import transformers;print(transformers.__version__)"', cwd=str(ROOT)),
            "hub": ver('.venv_h3/bin/python -c "import huggingface_hub;print(huggingface_hub.__version__)"', cwd=str(ROOT)),
        },
        "system_python_versions": {
            "why_recorded": "E8-E18 ran here; the two sets of versions are incompatible by design",
            "torch": ver('python3 -c "import torch;print(torch.__version__)"', cwd=str(ROOT)),
            "diffusers": ver('python3 -c "import diffusers;print(diffusers.__version__)"', cwd=str(ROOT)),
        },
        "gpu": "4x NVIDIA B200 (183 GB each)",
        "peak_vram_gb": {"real": 119.8, "random": 133.4,
                         "note": "one process per GPU; two would OOM"},
    },
    "scripts": {p: sha(p) for p in sorted(glob.glob("scripts/e19_*.py"))
                + sorted(glob.glob("scripts/e19_*.sh"))
                + ["scripts/piq_par_download.py"]},
    "download": {"tool": "scripts/piq_par_download.py",
                 "why": ("single-stream throughput to HF's Xet CDN through the proxy was 65 KB/s "
                         "(120 GB would take ~3 weeks). Measured scaling: 8 streams 0.7 MB/s, "
                         "32 -> 2.7, 96 -> 5.5-6.3. Chunked range requests with resume."),
                 "throughput_mb_s": 6.3},
    "deliverables": ["E19_PREREG.md", "E19_cells.json", "E19_verdict.json", "E19_REPORT.md",
                     "manifest.json"],
}

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "manifest.json").write_text(json.dumps(m, indent=1))
print(json.dumps(m["extraction_counts"], indent=1))
print("sizes:", m["sizes_gb"])
print("e19 env:", m["env"]["e19_versions"])
print("wrote", OUT / "manifest.json")
