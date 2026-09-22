"""Shared Wan2.2-TI2V-5B probing primitives (T2 minimal chain, T3 null floor, T4 CALIPER).

All numerical conventions follow research/MODELS.md, with two corrections found on the box:

* `UniPCMultistepScheduler` in diffusers 0.36 has **no** `scale_noise`; only the FlowMatch*
  schedulers do. We therefore do the flow-matching interpolation ourselves:
      sigma  = t / num_train_timesteps          (t is the integer timestep fed to the DiT)
      z_t    = (1 - sigma) * z_0 + sigma * eps
      v_tgt  = eps - z_0                        (MODELS.md risk #5)
* Wan2.2-TI2V needs a **per-token** timestep of shape [B, seq_len] (`expand_timesteps=true`,
  MODELS.md risk #3). A scalar broadcasts silently but is semantically wrong.

Latents are normalised with the per-channel `latents_mean` / `latents_std` (risk #1).
The 11 GB UMT5 text encoder is never loaded here; prompt embeds come from disk (risk #8).
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
REVISION = "b8fff7315c768468a5333511427288870b2e9635"
NUM_TRAIN_TIMESTEPS = 1000
PATCH = (1, 2, 2)


# --------------------------------------------------------------------------------------
# model loading
# --------------------------------------------------------------------------------------
def snapshot_dir(repo: str = REPO, revision: str = REVISION) -> Path:
    """Local snapshot path inside $HF_HOME/hub.

    Submodules are loaded from the local path rather than the repo id: for sharded
    checkpoints diffusers calls the Hub `model_info` API even when everything is cached,
    which fails on a box without direct egress and makes runs depend on the network.
    """
    import os

    d = Path(os.environ["HF_HOME"]) / "hub" / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
    if not d.exists():
        raise FileNotFoundError(f"{d} missing — run scripts/t2_download_wan22.py first")
    return d


def load_vae(device: str = "cuda:0", dtype: torch.dtype = torch.float32):
    from diffusers import AutoencoderKLWan

    vae = AutoencoderKLWan.from_pretrained(str(snapshot_dir() / "vae"), torch_dtype=dtype)
    return vae.to(device).eval()


def load_transformer(device: str = "cuda:0", dtype: torch.dtype = torch.bfloat16, random_init: bool = False):
    """Load the DiT. `random_init=True` gives the CALIPER lower-bound backbone (E1)."""
    from diffusers import WanTransformer3DModel

    src = str(snapshot_dir() / "transformer")
    if random_init:
        cfg = WanTransformer3DModel.load_config(src)
        with torch.device("meta"):
            model = WanTransformer3DModel.from_config(cfg)
        model = model.to_empty(device=device)
        _init_weights_(model)
        return model.to(dtype).eval()

    model = WanTransformer3DModel.from_pretrained(src, torch_dtype=dtype)
    return model.to(device).eval()


@torch.no_grad()
def _init_weights_(model: torch.nn.Module, seed: int = 0) -> None:
    """Deterministic re-initialisation for the random-init control backbone."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    for name, p in model.named_parameters():
        if p.ndim >= 2:
            fan_in = int(np.prod(p.shape[1:]))
            std = (2.0 / max(fan_in, 1)) ** 0.5
            p.copy_(torch.empty(p.shape, dtype=torch.float32).normal_(0.0, std, generator=g).to(p.dtype))
        else:
            p.zero_()
    for name, b in model.named_buffers():
        if b.is_floating_point():
            b.zero_()


def load_prompt_embeds(path: str | Path, device: str = "cuda:0", dtype: torch.dtype = torch.bfloat16):
    blob = torch.load(path, map_location="cpu")
    return blob["embeds"].to(device=device, dtype=dtype), blob


# --------------------------------------------------------------------------------------
# video -> latents
# --------------------------------------------------------------------------------------
def read_video_frames(path: str | Path, num_frames: int, stride: int = 1, start: int = 0) -> np.ndarray:
    """Decode `num_frames` frames (uint8, [F,H,W,3]). Raises if the clip is too short."""
    import decord

    vr = decord.VideoReader(str(path))
    idx = [start + i * stride for i in range(num_frames)]
    if idx[-1] >= len(vr):
        raise ValueError(f"{path}: needs frame {idx[-1]} but clip has {len(vr)}")
    return vr.get_batch(idx).asnumpy()


def preprocess_frames(frames: np.ndarray, height: int, width: int, device: str = "cuda:0") -> torch.Tensor:
    """uint8 [F,H,W,3] -> float32 [1,3,F,H,W] in [-1,1], bilinear-resized to (height,width)."""
    assert height % 32 == 0 and width % 32 == 0, "TI2V-5B needs H/W divisible by 32 (risk #10)"
    x = torch.from_numpy(frames).to(device=device, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    if x.shape[-2:] != (height, width):
        x = torch.nn.functional.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)
    x = x * 2.0 - 1.0
    return x.permute(1, 0, 2, 3).unsqueeze(0).contiguous()


@torch.no_grad()
def encode_latents(vae, pixels: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """[1,3,F,H,W] in [-1,1] -> latents [1,z,F',h,w]; `normalize` applies per-channel mean/std."""
    posterior = vae.encode(pixels.to(next(vae.parameters()).dtype)).latent_dist
    z = posterior.mode()
    if not normalize:
        return z.float()
    m = torch.tensor(vae.config.latents_mean, device=z.device, dtype=torch.float32).view(1, -1, 1, 1, 1)
    s = torch.tensor(vae.config.latents_std, device=z.device, dtype=torch.float32).view(1, -1, 1, 1, 1)
    return (z.float() - m) / s


# --------------------------------------------------------------------------------------
# flow-matching noising
# --------------------------------------------------------------------------------------
def flow_noise(z0: torch.Tensor, t_int: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (z_t, eps, v_target) with sigma = t/1000 and v_target = eps - z0 (risk #5)."""
    g = torch.Generator(device=z0.device).manual_seed(int(seed))
    eps = torch.randn(z0.shape, generator=g, device=z0.device, dtype=torch.float32)
    sigma = float(t_int) / NUM_TRAIN_TIMESTEPS
    z_t = (1.0 - sigma) * z0 + sigma * eps
    return z_t, eps, eps - z0


def per_token_timestep(z_t: torch.Tensor, t_int: int) -> torch.Tensor:
    """[B, seq_len] per-token timestep for expand_timesteps=True (risk #3)."""
    b, _, f, h, w = z_t.shape
    ph, pw = PATCH[1], PATCH[2]
    ones = torch.ones((f, h, w), device=z_t.device, dtype=torch.float32)
    temp = (ones[:, ::ph, ::pw] * float(t_int)).flatten()
    return temp.unsqueeze(0).expand(b, -1)


def latent_grid(z_t: torch.Tensor) -> tuple[int, int, int]:
    """Post-patch (frames, height, width) token grid."""
    _, _, f, h, w = z_t.shape
    return f // PATCH[0], h // PATCH[1], w // PATCH[2]


# --------------------------------------------------------------------------------------
# hidden-state capture
# --------------------------------------------------------------------------------------
@dataclass
class HiddenStates:
    """Per-frame pooled features for each requested block: {layer: [F', D] float32 cpu}."""

    pooled: dict[int, torch.Tensor] = field(default_factory=dict)
    grid: tuple[int, int, int] | None = None


@contextlib.contextmanager
def capture_blocks(transformer, layers: list[int], grid: tuple[int, int, int]):
    """Hook `transformer.blocks[i]` outputs and mean-pool tokens spatially per latent frame.

    Wan block hidden states are video-only ([B, seq, D]); text lives in encoder_hidden_states,
    so unlike CogVideoX there are no text tokens to strip.
    """
    store = HiddenStates(grid=grid)
    f, hh, ww = grid
    handles = []

    def make_hook(idx: int):
        def hook(_module, _inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            b, seq, d = h.shape
            assert seq == f * hh * ww, f"seq {seq} != {f}*{hh}*{ww}"
            pooled = h.float().view(b, f, hh * ww, d).mean(dim=2)[0]
            store.pooled[idx] = pooled.detach().to("cpu")

        return hook

    try:
        for i in layers:
            handles.append(transformer.blocks[i].register_forward_hook(make_hook(i)))
        yield store
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def forward_v(transformer, z_t: torch.Tensor, timestep: torch.Tensor, prompt_embeds: torch.Tensor) -> torch.Tensor:
    # NB: `next(transformer.parameters()).dtype` is wrong here — diffusers keeps
    # `_keep_in_fp32_modules` (time_embedder, scale_shift_table, norms) in fp32 even when the
    # model is loaded in bf16, so the first parameter may be fp32 while the conv is bf16.
    dtype = transformer.patch_embedding.weight.dtype
    out = transformer(
        hidden_states=z_t.to(dtype),
        timestep=timestep,
        encoder_hidden_states=prompt_embeds.to(dtype),
        return_dict=False,
    )[0]
    return out.float()


def denoising_error(v_pred: torch.Tensor, v_target: torch.Tensor) -> dict:
    err = (v_pred - v_target) ** 2
    return {
        "mse": err.mean().item(),
        "per_frame": err.mean(dim=(0, 1, 3, 4)).cpu().tolist(),
    }


# --------------------------------------------------------------------------------------
# misc
# --------------------------------------------------------------------------------------
def preflight(device: str = "cuda:0", n: int = 1 << 24, strict: bool = True) -> dict:
    """Detect a compute-faulty GPU before any result is written.

    Box reality (2026-09-16): physical GPU 0 (PCI 0000:1b:00) has a defective SM cluster.
    `dmesg` shows `NVRM: Xid 13 ... Graphics SM Warp Exception ... Illegal Instruction
    Encoding` on GPC 10, and a trivial `x*2+1` kernel returns ~8k wrong elements out of
    16.7M, with errors up to 2.4e3. Host<->device DMA is fine, so nothing crashes and the
    numbers just come out wrong. Every entry point must call this first.
    """
    x = torch.arange(n, dtype=torch.float32) % 1000
    y = (x.to(device) * 2.0 + 1.0).cpu()
    d = (y - (x * 2.0 + 1.0)).abs()
    out = {
        "device": device,
        "physical_device": _physical_device(device),
        "n": n,
        "max_abs_diff": d.max().item(),
        "n_wrong": int((d > 0).sum()),
    }
    out["healthy"] = out["n_wrong"] == 0
    if strict and not out["healthy"]:
        raise RuntimeError(
            f"GPU compute self-check FAILED on {device} "
            f"(physical {out['physical_device']}): {out['n_wrong']} wrong elements, "
            f"max|diff|={out['max_abs_diff']:.4g}. Do not trust any number from this device."
        )
    return out


def _physical_device(device: str) -> str:
    import os

    idx = int(device.split(":")[1]) if ":" in device else 0
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis:
        parts = [p.strip() for p in vis.split(",") if p.strip() != ""]
        if idx < len(parts):
            return f"physical cuda:{parts[idx]} (CUDA_VISIBLE_DEVICES={vis})"
    return f"physical cuda:{idx} (CUDA_VISIBLE_DEVICES unset)"


def vram_report(tag: str = "") -> dict:
    return {
        "tag": tag,
        "allocated_GB": round(torch.cuda.memory_allocated() / 2**30, 3),
        "max_allocated_GB": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "reserved_GB": round(torch.cuda.memory_reserved() / 2**30, 3),
    }


def env_stamp() -> dict:
    """Provenance block that must accompany every result file (HANDOFF engineering rule)."""
    import platform
    import subprocess
    import sys

    import diffusers
    import transformers

    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        rev = "no-git"
    return {
        "python": sys.version.split()[0],
        "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "repo": REPO,
        "revision": REVISION,
        "code_rev": rev,
    }
