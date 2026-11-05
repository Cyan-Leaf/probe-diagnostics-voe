"""Shared CogVideoX-2b probing primitives for E8 (second backbone).

E8 asks whether the D1/D2 verdicts are a property of Wan2.2 or a property of video
diffusion models in general.  This module mirrors `wan_probe_lib` so the two backbones are
probed the same way, and deviates only where CogVideoX's architecture forces it.  The
deviations are listed in results/E8_second_backbone/E8_PREREG.md, written before extraction.

Why CogVideoX-2b: `num_layers = 30`, the same as Wan2.2-TI2V-5B, so the frozen reference
cell "layer 16" is the same relative depth (16/30) on both.  LTX-Video has 28 layers and
would have moved the cell.

The four places CogVideoX is not Wan:

1. Noising.  Wan is flow matching (`z_t = (1-σ)z₀ + σε`, σ = t/1000).  CogVideoX is VP/DDIM
   with `rescale_betas_zero_snr=True` and `snr_shift_scale=3.0`, so σ_eff(t) = sqrt(1-ᾱ_t)
   rises much faster: at t=600 Wan sits at σ=0.60 while CogVideoX sits at σ_eff=0.977.
   `noise_schedule()` below returns the same table the pre-registration quotes, and
   `snr_matched_t()` gives the CogVideoX timestep whose SNR equals a Wan one.  Both the
   literal t=600 cell and the SNR-matched t=298 cell are extracted; neither is preferred
   after seeing a result.

2. Random-init control.  `wan_probe_lib._init_weights_` zeroes every 1-D parameter,
   including RMSNorm weights — harmless on Wan because its modulation lives in the 2-D
   `scale_shift_table`.  On CogVideoX the affine norm weight *is* the normaliser
   (`CogVideoXLayerNormZero`), so zeroing it makes `norm(x) = 0` and every block collapse
   to `x + gate*W·shift`, independent of the input.  Measured: between-clip feature std
   exactly 0.000000.  So the random control here keeps norm weights at 1.0 — see
   `_init_weights_norm_aware`.

3. Hidden states.  Wan blocks are video-only.  CogVideoX's patch embed concatenates text
   tokens in front of the video tokens, but `CogVideoXTransformer3DModel.forward` splits
   them off again (`hidden_states = hidden_states[:, text_seq_length:]`) *before* the block
   loop, so `output[0]` from a block hook is already video-only — same as Wan.  The shape
   assertion below is what enforces that.

4. VAE normalisation.  `AutoencoderKLCogVideoX` has `latents_mean = latents_std = null` and
   a single scalar `scaling_factor` (1.15258426).  Wan normalises per channel with a 48-vector.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

REPO = "zai-org/CogVideoX-2b"
REVISION = "1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01"
NUM_TRAIN_TIMESTEPS = 1000
TEXT_SEQ_LEN = 226

# Pre-registered cells.  T_LITERAL matches PREREG.md's reference cell verbatim; T_SNR is the
# timestep whose noise level equals Wan's t=600 (see E8_PREREG.md §2).
T_LITERAL = 600
T_SNR = 298


# --------------------------------------------------------------------------------------
# model loading
# --------------------------------------------------------------------------------------
def snapshot_dir(repo: str = REPO, revision: str = REVISION) -> Path:
    """Local snapshot path inside $HF_HOME/hub.

    Same rationale as wan_probe_lib.snapshot_dir: loading submodules from the local path
    avoids diffusers' Hub `model_info` call, which makes runs depend on egress.
    """
    d = Path(os.environ["HF_HOME"]) / "hub" / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
    if not d.exists():
        raise FileNotFoundError(f"{d} missing — download {repo} first")
    return d


def load_vae(device: str = "cuda:0", dtype: torch.dtype = torch.float32):
    from diffusers import AutoencoderKLCogVideoX

    vae = AutoencoderKLCogVideoX.from_pretrained(str(snapshot_dir() / "vae"), torch_dtype=dtype)
    return vae.to(device).eval()


def load_transformer(device: str = "cuda:0", dtype: torch.dtype = torch.bfloat16,
                     random_init: bool = False):
    """Load the DiT. `random_init=True` gives the E8 lower-bound backbone."""
    from diffusers import CogVideoXTransformer3DModel

    src = str(snapshot_dir() / "transformer")
    if random_init:
        cfg = CogVideoXTransformer3DModel.load_config(src)
        with torch.device("meta"):
            model = CogVideoXTransformer3DModel.from_config(cfg)
        model = model.to_empty(device=device)
        _init_weights_norm_aware(model)
        return model.to(dtype).eval()

    model = CogVideoXTransformer3DModel.from_pretrained(src, torch_dtype=dtype)
    return model.to(device).eval()


@torch.no_grad()
def _init_weights_norm_aware(model: torch.nn.Module, seed: int = 0) -> None:
    """Deterministic re-init for the random control, adapted to CogVideoX's affine norms.

    Differs from `wan_probe_lib._init_weights_` in two respects, both forced:

    * `norm.weight` gets 1.0 instead of 0.0.  CogVideoXLayerNormZero does
      `self.norm(x) * (1 + scale) + shift`; with weight 0 the first term vanishes and the
      block output no longer depends on the video at all.  Verified empirically: the Wan
      convention gives between-clip feature std exactly 0.
    * Floating buffers (`patch_embedding.pos_embedding`) get N(0, 0.02) instead of 0.
      CogVideoX-2b has `use_rotary_positional_embeddings=false`, so this buffer is the only
      source of position information; zeroing it removes spatial structure.

    Everything else follows the Wan rule so the two random backbones are as alike as their
    architectures allow.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    for name, p in model.named_parameters():
        if p.ndim >= 2:
            fan_in = int(np.prod(p.shape[1:]))
            std = (2.0 / max(fan_in, 1)) ** 0.5
            p.copy_(torch.empty(p.shape, dtype=torch.float32).normal_(0.0, std, generator=g).to(p.dtype))
        elif name.endswith("norm.weight"):
            p.fill_(1.0)
        else:
            p.zero_()
    for name, b in model.named_buffers():
        if b.is_floating_point():
            b.copy_(torch.empty(b.shape, dtype=torch.float32).normal_(0.0, 0.02, generator=g).to(b.dtype))


def load_text_embeds(path: str | Path, device: str = "cuda:0", dtype: torch.dtype = torch.bfloat16):
    """Cached empty-prompt T5 embedding, [1, 226, 4096]."""
    blob = torch.load(path, map_location="cpu")
    return blob["embeds"].to(device=device, dtype=dtype), blob


@torch.no_grad()
def build_text_embeds(out_path: str | Path, prompt: str = "",
                      device: str = "cuda:0") -> dict:
    """One-off: encode a prompt with the repo's own T5 (v1_1-XXL) and cache it.

    The text encoder is 9.5 GB in fp16 and identical for every clip, so it is loaded once
    here and never again — same treatment Wan's UMT5 got (MODELS.md risk #8).

    Environment note: this box has a broken `apex` (its `fused_layer_norm_cuda` extension is
    absent).  `transformers.models.t5.modeling_t5` does `from apex.normalization import
    FusedRMSNorm` at import time and rebinds `T5LayerNorm` to it; the failure only surfaces
    later, at `T5LayerNorm.__init__`, so it looks like an unrelated crash deep inside the
    model.  `_block_apex()` forces the pure-PyTorch T5LayerNorm instead.  This changes no
    number: FusedRMSNorm is a drop-in for the same normalisation.
    """
    _block_apex()
    from transformers.models.t5.modeling_t5 import T5EncoderModel
    from transformers import T5Tokenizer

    d = snapshot_dir()
    tok = T5Tokenizer.from_pretrained(str(d / "tokenizer"))
    te = T5EncoderModel.from_pretrained(str(d / "text_encoder"), torch_dtype=torch.float16).to(device).eval()
    ids = tok(prompt, padding="max_length", max_length=TEXT_SEQ_LEN, truncation=True,
              add_special_tokens=True, return_tensors="pt").input_ids.to(device)
    embeds = te(ids)[0]
    blob = {"prompts": [prompt], "embeds": embeds.cpu().half(), "seq_lens": ids.ne(0).sum(1).cpu(),
            "repo": REPO, "revision": REVISION, "max_sequence_length": TEXT_SEQ_LEN,
            "note": "encoded with T5LayerNorm fallback (apex blocked); see build_text_embeds"}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, out_path)
    del te
    torch.cuda.empty_cache()
    return blob


def _block_apex() -> None:
    """Make `from apex.normalization import FusedRMSNorm` raise ImportError, so that
    `transformers.models.t5.modeling_t5` keeps its own pure-PyTorch `T5LayerNorm`.

    Idempotent.  The installed apex has no `fused_layer_norm_cuda` extension, and
    modeling_t5 binds `T5LayerNorm = FusedRMSNorm` inside a `try/except ImportError` at
    import time; because the import itself succeeds, the missing extension only explodes
    later, inside `T5LayerNorm.__init__`, which reads as an unrelated crash deep in the model.

    Blocks `apex.normalization` specifically and not `apex`: transformers' own
    `import_utils` probes the top-level `apex` with `importlib.util.find_spec()` during
    import, and a finder that raises for that name breaks it.

    Changes no number — FusedRMSNorm and T5LayerNorm implement the same normalisation; the
    recorded `note` in the cached embeddings says which one was used.
    """
    import importlib
    import importlib.abc
    import sys

    class _ApexNormBlocker(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "apex.normalization" or fullname.startswith("apex.normalization."):
                raise ImportError("apex.normalization blocked: fused_layer_norm_cuda missing")
            return None

    if not any("_ApexNormBlocker" in repr(type(f)) for f in sys.meta_path):
        sys.meta_path.insert(0, _ApexNormBlocker())

    for mod in [m for m in sys.modules if m == "apex.normalization" or m.startswith("apex.normalization.")]:
        del sys.modules[mod]

    # modeling_t5 caches T5LayerNorm at import time, so it must be re-imported for the
    # fallback to take effect.
    for mod in [m for m in sys.modules if m == "transformers.models.t5.modeling_t5"]:
        del sys.modules[mod]
    if "transformers.models.t5" in sys.modules:
        importlib.reload(sys.modules["transformers.models.t5"])


# --------------------------------------------------------------------------------------
# noise schedule (the part that makes t=600 mean different things)
# --------------------------------------------------------------------------------------
def scheduler():
    from diffusers import CogVideoXDDIMScheduler

    return CogVideoXDDIMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
        num_train_timesteps=NUM_TRAIN_TIMESTEPS, prediction_type="v_prediction",
        rescale_betas_zero_snr=True, snr_shift_scale=3.0, set_alpha_to_one=True,
        clip_sample=False, timestep_spacing="trailing")


def noise_schedule() -> dict:
    """ᾱ_t / σ_eff(t) / SNR(t), plus the Wan-equivalent timestep for σ = t/1000."""
    ab = scheduler().alphas_cumprod.numpy()
    snr = ab / (1.0 - ab)
    wan_snr = (0.4 ** 2) / (0.6 ** 2)          # Wan t=600: 1-σ = 1-0.6
    t_snr = int(np.argmin(np.abs(snr - wan_snr)))
    return {
        "alphas_cumprod": ab,
        "t_literal": T_LITERAL,
        "t_literal_alpha_bar": float(ab[T_LITERAL]),
        "t_literal_sigma_eff": float(np.sqrt(1 - ab[T_LITERAL])),
        "t_literal_snr": float(snr[T_LITERAL]),
        "wan_t600_sigma": 0.6,
        "wan_t600_snr": float(wan_snr),
        "t_snr_matched": t_snr,
        "t_snr_matched_alpha_bar": float(ab[t_snr]),
        "t_snr_matched_sigma_eff": float(np.sqrt(1 - ab[t_snr])),
        "t_snr_matched_snr": float(snr[t_snr]),
        "table": {str(t): {"alpha_bar": float(ab[t]), "sigma_eff": float(np.sqrt(1 - ab[t])),
                           "snr": float(snr[t])}
                  for t in (0, 100, 200, 298, 400, 600, 800, 999)},
    }


def flow_noise(z0: torch.Tensor, t_int: int, seed: int):
    """VP noising: returns (z_t, eps, v_target) with v_target = √ᾱ·ε − √(1−ᾱ)·z₀.

    Mirrors the Wan helper's contract so downstream code is shared; only the interpolation
    differs, because CogVideoX predicts v under a VP schedule rather than flow matching.
    """
    ab = float(scheduler().alphas_cumprod[t_int])
    root_ab, root_1m = ab ** 0.5, (1.0 - ab) ** 0.5
    g = torch.Generator(device=z0.device).manual_seed(int(seed))
    eps = torch.randn(z0.shape, generator=g, device=z0.device, dtype=torch.float32)
    z_t = (root_ab * z0 + root_1m * eps).to(z0.dtype)
    return z_t, eps, root_ab * eps - root_1m * z0


# --------------------------------------------------------------------------------------
# video -> latents
# --------------------------------------------------------------------------------------
def read_spanning(path: str | Path, n: int, size: int = 512) -> np.ndarray:
    """n frames spanning the whole clip. Same rule as E7: the violation is not annotated,
    so the first 49 native frames would cover 8% of the clip and could miss it entirely."""
    import decord

    vr = decord.VideoReader(str(path), width=size, height=size)
    idx = np.linspace(0, len(vr) - 1, n).round().astype(int).tolist()
    return vr.get_batch(idx).asnumpy()


def preprocess_frames(frames: np.ndarray, height: int, width: int,
                      device: str = "cuda:0") -> torch.Tensor:
    """uint8 [F,H,W,3] -> float32 [1,3,F,H,W] in [-1,1]."""
    assert height % 8 == 0 and width % 8 == 0, "CogVideoX VAE downsamples 8x spatially"
    assert (frames.shape[0] - 1) % 4 == 0, "CogVideoX VAE downsamples 4x temporally"
    x = torch.from_numpy(frames).to(device=device, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    if x.shape[-2:] != (height, width):
        x = torch.nn.functional.interpolate(x, size=(height, width), mode="bilinear",
                                            align_corners=False)
    x = x * 2.0 - 1.0
    return x.permute(1, 0, 2, 3).unsqueeze(0).contiguous()


@torch.no_grad()
def encode_latents(vae, pixels: torch.Tensor) -> torch.Tensor:
    """[1,3,F,H,W] in [-1,1] -> latents [1,F',16,h,w], scaled by the VAE's scaling_factor.

    Returned in the transformer's layout, not the VAE's.  `AutoencoderKLCogVideoX` emits
    [B, C, F', h, w] whereas `CogVideoXTransformer3DModel.forward` unpacks
    `batch_size, num_frames, channels, height, width = hidden_states.shape` — i.e. it wants
    [B, F', C, h, w].  Feeding the VAE layout straight in fails at `patch_embed.proj` with
    "expected input[16, 13, 64, 64] to have 16 channels".  Converting once here means every
    downstream helper (noising, pooling, grid) works in a single frame-major layout, the same
    one Wan uses.

    CogVideoX has no per-channel latents_mean/std; `scaling_factor` is a single scalar and
    is the whole normalisation.
    """
    posterior = vae.encode(pixels.to(next(vae.parameters()).dtype)).latent_dist
    z = posterior.mode().float()
    return (z * float(vae.config.scaling_factor)).permute(0, 2, 1, 3, 4).contiguous()


def latent_grid(z_t: torch.Tensor) -> tuple[int, int, int]:
    """Post-patch (frames, height, width) token grid. patch_size_t is None for 2b, so the
    frame count is unchanged and only H/W are halved by patch_size=2.

    `z_t` is [B, F', C, h, w].
    """
    _, f, _, h, w = z_t.shape
    p = 2
    return f, h // p, w // p


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
    """Hook `transformer.transformer_blocks[i]` and mean-pool tokens spatially per frame.

    Same pooling as Wan (mean over the h*w tokens of each latent frame), so the feature
    geometry is identical: [F', D] per layer, F'=13 here.

    CogVideoX blocks return `(hidden_states, encoder_hidden_states)`.  `forward` splits text
    tokens off *before* the block loop, so output[0] is video-only — the assertion on the
    token count is what would catch a diffusers version that changed that.
    """
    store = HiddenStates(grid=grid)
    f, hh, ww = grid
    handles = []

    def make_hook(idx: int):
        def hook(_module, _inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            b, seq, d = h.shape
            assert seq == f * hh * ww, f"seq {seq} != {f}*{hh}*{ww} (text tokens not stripped?)"
            pooled = h.float().view(b, f, hh * ww, d).mean(dim=2)[0]
            store.pooled[idx] = pooled.detach().to("cpu")

        return hook

    try:
        for i in layers:
            handles.append(transformer.transformer_blocks[i].register_forward_hook(make_hook(i)))
        yield store
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def _rotary_embeds(transformer, grid: tuple[int, int, int], device) -> tuple | None:
    """3-D RoPE for models that need it (CogVideoX-5b has `use_rotary_positional_embeddings`).

    Reproduces `CogVideoXPipeline._prepare_rotary_positional_embeddings` for the
    `patch_size_t is None` case, which covers CogVideoX 1.0 (2b and 5b).  Returned as a
    plain tuple so callers can pass it straight into `forward`.

    This is the one place where 2b and 5b genuinely diverge: 2b learns a positional
    embedding table inside `patch_embed`, 5b uses RoPE passed in from outside.  Omitting it
    for 5b does not raise — the model runs, but every token is position-blind, and the
    physics probe collapses to chance while the appearance control still works (colour
    needs no position).  That failure mode is silent and it produced a whole wrong run.
    """
    if not getattr(transformer.config, "use_rotary_positional_embeddings", False):
        return None
    from diffusers.models.embeddings import get_3d_rotary_pos_embed
    from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid

    f, hh, ww = grid
    p = transformer.config.patch_size
    base_w = transformer.config.sample_width // p
    base_h = transformer.config.sample_height // p
    crops = get_resize_crop_region_for_grid((hh, ww), base_w, base_h)
    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=transformer.config.attention_head_dim,
        crops_coords=crops,
        grid_size=(hh, ww),
        temporal_size=f,
        device=device,
    )
    return freqs_cos, freqs_sin


@torch.no_grad()
def forward_v(transformer, z_t: torch.Tensor, timestep, prompt_embeds: torch.Tensor,
              grid: tuple[int, int, int] | None = None) -> torch.Tensor:
    """Velocity prediction. `grid` is required for RoPE models (see `_rotary_embeds`)."""
    dtype = transformer.patch_embed.proj.weight.dtype
    kwargs = {}
    if getattr(transformer.config, "use_rotary_positional_embeddings", False):
        if grid is None:
            raise ValueError(
                f"{transformer.__class__.__name__} uses rotary embeddings; pass the token "
                f"grid so the position encoding can be built (see _rotary_embeds)")
        kwargs["image_rotary_emb"] = _rotary_embeds(transformer, grid, z_t.device)
    out = transformer(
        hidden_states=z_t.to(dtype),
        encoder_hidden_states=prompt_embeds.to(dtype),
        timestep=timestep,
        return_dict=False,
        **kwargs,
    )[0]
    return out.float()


def pooled_raw_pixel(px: torch.Tensor, size: int = 32) -> torch.Tensor:
    """[1,3,F,H,W] in [-1,1] -> per-latent-frame 32x32 RGB, flattened. Identical to E7's."""
    x = (px[0].permute(1, 0, 2, 3) + 1.0) * 0.5
    x = torch.nn.functional.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    groups = [x[0:1]] + [x[1 + 4 * i: 1 + 4 * (i + 1)] for i in range((x.shape[0] - 1) // 4)]
    return torch.stack([g.mean(dim=0) for g in groups], dim=0).flatten(1).half().cpu()
