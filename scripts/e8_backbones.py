"""E8 backbone adapters: one interface over several video-diT families.

The four lower bounds and the D1/D2 probe want the same thing from every model: per-frame
pooled hidden states at a chosen block, VAE latents, and a way to noise the latents.  This
module gives that, and isolates the places where the families genuinely differ.

Two axes matter and they are deliberately kept separate:

  family   Wan / CogVideoX — different noising (flow matching vs VP), different VAE
           (16 or 48 channels vs 16), different block layout (`blocks` vs
           `transformer_blocks`), different text stack
  scale    within Wan: 1.3B / 5B / 14B — same code path, different depth

Why the Wan ladder is not a clean size axis, stated up front because it affects how the
result must be read:

  1.3B and 14B are T2V models with a **16-channel** `AutoencoderKLWan`.
  TI2V-5B is a different VAE entirely — 48 channels, causal, z_dim 48.
  So moving 1.3B -> 14B holds the VAE fixed and changes only the transformer;
  moving 14B -> the E7 reference (5B) changes BOTH.  Any 5B-vs-14B difference therefore
  cannot be attributed to size alone, and the report must say so.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

import cogvideox_probe_lib as C
import wan_probe_lib as W


@dataclass(frozen=True)
class Spec:
    key: str            # registry key used in the analyse/verdict scripts
    label: str
    repo: str
    family: str         # "wan" | "cogvideox"
    params_b: float
    n_layers: int
    latent_channels: int
    block_attr: str     # attribute holding the transformer blocks
    schedule: str       # "flow" | "vp"
    revision: str | None = None


REGISTRY: dict[str, Spec] = {
    "wan21_1b3": Spec("wan21_1b3", "Wan2.1-T2V-1.3B",
                      "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "wan", 1.3, 30, 16,
                      "blocks", "flow", "0fad780a534b6463e45facd96134c9f345acfa5b"),
    "wan22_5b": Spec("wan22_5b", "Wan2.2-TI2V-5B (E7 reference)",
                     "Wan-AI/Wan2.2-TI2V-5B-Diffusers", "wan", 5.0, 30, 48,
                     "blocks", "flow", W.REVISION),
    "wan21_14b": Spec("wan21_14b", "Wan2.1-T2V-14B",
                      "Wan-AI/Wan2.1-T2V-14B-Diffusers", "wan", 14.0, 40, 16,
                      "blocks", "flow", None),
    "cogvideox2b": Spec("cogvideox2b", "CogVideoX-2b", C.REPO, "cogvideox", 2.0, 30, 16,
                        "transformer_blocks", "vp", C.REVISION),
    "cogvideox5b": Spec("cogvideox5b", "CogVideoX-5b", "zai-org/CogVideoX-5b", "cogvideox",
                        5.0, 42, 16, "transformer_blocks", "vp",
                        "8fc5b281006c82b82d34fd2543d2f0ebb4e7e321"),
}


def snapshot(spec: Spec) -> Path:
    """Local snapshot dir for a spec; Wan's revision is optional (any downloaded one)."""
    import os

    base = Path(os.environ["HF_HOME"]) / "hub" / ("models--" + spec.repo.replace("/", "--")) / "snapshots"
    if spec.revision:
        d = base / spec.revision
        if d.exists():
            return d
    cands = sorted(p for p in base.glob("*") if p.is_dir())
    if not cands:
        raise FileNotFoundError(f"{base} has no snapshot — download {spec.repo} first")
    return cands[-1]


def load_vae(spec: Spec, device: str = "cuda:0", dtype=torch.float32):
    if spec.family == "wan":
        from diffusers import AutoencoderKLWan

        vae = AutoencoderKLWan.from_pretrained(str(snapshot(spec) / "vae"), torch_dtype=dtype)
    else:
        from diffusers import AutoencoderKLCogVideoX

        vae = AutoencoderKLCogVideoX.from_pretrained(str(snapshot(spec) / "vae"),
                                                     torch_dtype=dtype)
    return vae.to(device).eval()


def load_transformer(spec: Spec, device: str = "cuda:0", dtype=torch.bfloat16,
                     random_init: bool = False):
    """Wan and CogVideoX take the same call shape; only the class and the init differ."""
    if spec.family == "wan":
        from diffusers import WanTransformer3DModel as Cls
        init_fn = W._init_weights_
    else:
        from diffusers import CogVideoXTransformer3DModel as Cls
        init_fn = C._init_weights_norm_aware

    src = str(snapshot(spec) / "transformer")
    if random_init:
        cfg = Cls.load_config(src)
        with torch.device("meta"):
            model = Cls.from_config(cfg)
        model = model.to_empty(device=device)
        init_fn(model)
        return model.to(dtype).eval()
    return Cls.from_pretrained(src, torch_dtype=dtype).to(device).eval()


def encode_latents(spec: Spec, vae, pixels: torch.Tensor) -> torch.Tensor:
    """[1,3,F,H,W] in [-1,1] -> transformer-layout latents.

    Wan's helper already returns [1, C, F', h, w]; CogVideoX's adapter converts to
    [1, F', C, h, w].  Both are passed through their own family's function so the
    per-family normalisation (per-channel mean/std for Wan, a scalar for CogVideoX) stays
    with the model that defines it.
    """
    if spec.family == "wan":
        return W.encode_latents(vae, pixels, normalize=True)
    return C.encode_latents(vae, pixels)


def latent_grid(spec: Spec, z_t: torch.Tensor) -> tuple[int, int, int]:
    if spec.family == "wan":
        return W.latent_grid(z_t)
    return C.latent_grid(z_t)


def noise(spec: Spec, z0: torch.Tensor, t_int: int, seed: int):
    """(z_t, v_target) under the family's own noising convention."""
    if spec.family == "wan":
        z_t, _eps, v = W.flow_noise(z0, t_int, seed=seed)
        return z_t, v
    z_t, _eps, v = C.flow_noise(z0, t_int, seed=seed)
    return z_t, v


def capture(spec: Spec, transformer, layers: list[int], grid: tuple[int, int, int]):
    if spec.family == "wan":
        return W.capture_blocks(transformer, layers, grid)
    return C.capture_blocks(transformer, layers, grid)


def forward(spec: Spec, transformer, z_t, prompt_embeds, t_int: int, grid):
    """Wan needs a per-token timestep; CogVideoX takes a scalar.

    `grid` is passed through because CogVideoX-5b builds RoPE from the token grid and gets
    no position information at all without it (2b does not — it has learned positional
    embeddings inside patch_embed).  See CogVideoX `_rotary_embeds`.
    """
    if spec.family == "wan":
        ts = W.per_token_timestep(z_t, t_int)
        return W.forward_v(transformer, z_t, ts, prompt_embeds)
    ts = torch.tensor([t_int], device=z_t.device)
    return C.forward_v(transformer, z_t, ts, prompt_embeds, grid=grid)


def text_embeds_path(spec: Spec) -> Path:
    """Cached empty-prompt embedding for this family (T5 vs UMT5 are not interchangeable)."""
    out = Path(__file__).resolve().parents[1] / "results" / "E8_second_backbone"
    if spec.family == "wan":
        return out / "prompt_embeds_empty_wan.pt"
    return out / "prompt_embeds_empty_cogvideox.pt"


def snr_matched_t(spec: Spec, wan_sigma: float = 0.6) -> int:
    """Timestep of THIS model whose SNR equals a Wan flow-matching t with that sigma.

    For a flow model that is just sigma*1000. For a VP model it is solved on the actual
    alphas_cumprod. Reported so a cross-family comparison states what it matched on.
    """
    import numpy as np

    if spec.family == "wan":
        return int(round(wan_sigma * 1000))
    ab = C.scheduler().alphas_cumprod.numpy()
    target = ((1 - wan_sigma) ** 2) / (wan_sigma ** 2)
    snr = ab / (1 - ab)
    return int(np.argmin(np.abs(snr - target)))
