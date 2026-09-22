"""Diagnose the VAE-encode CUDA failure (SDPA in AutoencoderKLWan mid-block attention).

Run: uv run --no-project python scripts/t2_diag_vae.py
"""

import json
import traceback

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

import wan_probe_lib as W

CASES = [
    ("fp32_default", torch.float32, None, False),
    ("fp32_math", torch.float32, SDPBackend.MATH, False),
    ("fp32_efficient", torch.float32, SDPBackend.EFFICIENT_ATTENTION, False),
    ("fp32_tiling", torch.float32, None, True),
    ("bf16_default", torch.bfloat16, None, False),
]


def main() -> None:
    frames = None
    out = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "cases": {}}
    for name, dtype, backend, tiling in CASES:
        try:
            vae = W.load_vae("cuda:0", dtype)
            if tiling:
                vae.enable_tiling(tile_sample_min_height=256, tile_sample_min_width=256)
            if frames is None:
                import glob
                import os

                hub = os.environ["HF_HOME"] + "/hub"
                vid = sorted(
                    glob.glob(hub + "/datasets--JianhaoDYDY--LikePhys-Benchmark/snapshots/*/data/"
                              "ball_drop_videos/subgroup_000/valid_00.mp4")
                )[0]
                frames = W.read_video_frames(vid, 49)
            px = W.preprocess_frames(frames, 512, 512, "cuda:0")
            ctx = sdpa_kernel(backend) if backend is not None else torch.no_grad()
            with torch.no_grad(), ctx:
                z = W.encode_latents(vae, px, normalize=True)
            torch.cuda.synchronize()
            out["cases"][name] = {"ok": True, "shape": list(z.shape),
                                 "mean": z.mean().item(), "std": z.std().item()}
            print(name, "OK", out["cases"][name], flush=True)
            del vae, z
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            out["cases"][name] = {"ok": False, "err": f"{type(exc).__name__}: {exc}".split("\n")[0]}
            print(name, "FAIL", out["cases"][name], flush=True)
            traceback.print_exc(limit=1)
            # a CUDA-level fault poisons the context; stop and report
            if "CUDA error" in str(exc):
                out["note"] = f"CUDA context poisoned at case {name}; rerun remaining cases separately"
                break
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
