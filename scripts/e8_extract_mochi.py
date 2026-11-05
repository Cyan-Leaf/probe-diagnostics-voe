"""E8 extraction for Mochi-1-preview (third family), separate from the generic path.

Why a separate script rather than another branch in `e8_extract_generic.py`: LTX differs
from Wan and CogVideoX in three structural ways that change the *extraction call*, not just
the weights.

  1. `MochiTransformer3DModel.forward` takes **flat tokens** `[B, T*H*W, C]`, plus
     `num_frames` / `height` / `width` as explicit arguments, because it builds its own RoPE
     internally from those three numbers (`self.rope(...)`).  There is no 5-D video tensor
     and no `image_rotary_emb` to pass in.  Forcing it through the generic path would have
     meant guessing the flattening order, which is exactly the class of mistake that
     silently destroys the signal (see E8_REPORT.md §12 for the CogVideoX-5b RoPE bug).
  2. `AutoencoderKLMochi` compresses **32x spatially** (measured: 256x256 -> 8x8) and
     **8x temporally** (measured: 49 -> 7), with 128 latent channels.  So one clip yields
     only 7*16*16 = 1792 tokens, versus 13312 for the other families.  The token count is a
     property of the model, and the report has to say so rather than pretend all backbones
     see the same grid.
  3. `AutoencoderKLMochi.forward()` runs encode **and** decode, so extraction must call
     `vae.encode()` directly; using `forward` would double the cost and return pixels.

Everything downstream (probe, D1, D2, floors) is untouched: the record layout is the same
as the other backbones, so `e8_analyse.py` reads it without modification.

Run:
  HF_HOME=... CUDA_VISIBLE_DEVICES=i python scripts/e8_extract_ltx.py \
      --dataset intphys2 --weights real --variant plain --shard i --num-shards 4
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cogvideox_probe_lib as C
import wan_probe_lib as W
from e8_extract import colour_jitter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E8_second_backbone"
N_FRAMES = 49
SIZE = 512
FEAT_SEEDS = [0, 1, 2]
REPO = "genmo/mochi-1-preview"

# Set in main() from the cached prompt-embedding blob; `forward_v` needs it to rebuild the
# encoder padding mask (Mochi cannot take encoder_attention_mask=None).
_SEQ_LENS = None
REVISION = "cd1d0fe89aa8382fcd9ae760bc3d7afe6183579f"

MANIFESTS = {
    "intphys2": ROOT / "results" / "E7_intphys2" / "manifest.json",
    "likephys": ROOT / "results" / "T3_null_floor" / "manifest.json",
}


def snapshot() -> Path:
    import os

    d = Path(os.environ["HF_HOME"]) / "hub" / ("models--" + REPO.replace("/", "--")) / "snapshots" / REVISION
    if not d.exists():
        raise FileNotFoundError(f"{d} missing")
    return d


@torch.no_grad()
def encode(vae, pixels: torch.Tensor) -> torch.Tensor:
    """[1,3,F,H,W] -> [1,F',C,h,w] (frame-major, like the other families' records).

    Mochi compresses 8x spatially and 6x temporally (measured), so 512x512x49 ->
    [1,9,12,64,64].  Stored frame-major for consistency; `forward_v` converts back because
    `MochiTransformer3DModel` unpacks `[B, C, F, H, W]`.
    """
    z = vae.encode(pixels.to(next(vae.parameters()).dtype)).latent_dist.mode()
    z = z.float() * float(vae.config.scaling_factor)
    return z.permute(0, 2, 1, 3, 4).contiguous()


def latent_grid(z: torch.Tensor) -> tuple[int, int, int]:
    """Post-patch token grid: Mochi has `patch_size=2`, so 64x64 latents -> 32x32 tokens.

    Unlike LTX (patch_size 1) the latent resolution is not the token resolution, and the
    block hooks see the patched token count.  Getting this wrong trips the `seq ==
    f*h*w` assertion in `capture` rather than silently pooling the wrong thing.
    """
    _, f, _, h, w = z.shape
    p = 2
    return f, h // p, w // p


@torch.no_grad()
def forward_v(tf, z_t: torch.Tensor, timestep, prompt_embeds: torch.Tensor,
              grid: tuple[int, int, int]) -> torch.Tensor:
    """Mochi takes the 5-D latent as `[B, C, F, H, W]` and builds its own RoPE internally.

    The record stores latents frame-major ([B, F, C, H, W]) for consistency with the other
    families, so this is where the permutation back happens.  Unlike LTX, no explicit
    num_frames/height/width arguments are needed.
    """
    hidden = z_t.permute(0, 2, 1, 3, 4).contiguous()      # [B, C, F, H, W]
    dtype = tf.patch_embed.proj.weight.dtype
    # Mochi's time_embed does `mask[:, None, None, :]` unconditionally, so None crashes
    # ("'NoneType' object is not subscriptable").
    #
    # The mask must come from the tokenizer, not from the embeddings: the cached T5 output
    # has non-zero values in the padding positions too (measured: 226/226 rows non-zero), so
    # "rows with non-zero norm" would mark every position valid and silently let the model
    # attend to padding.  The cache stores the real lengths, so rebuild the mask from them.
    seq_lens = _SEQ_LENS
    L = prompt_embeds.shape[1]
    mask = (torch.arange(L, device=prompt_embeds.device)[None, :]
            < seq_lens.to(prompt_embeds.device)[:, None]).to(torch.int64)
    out = tf(
        hidden_states=hidden.to(dtype),
        encoder_hidden_states=prompt_embeds.to(dtype),
        timestep=torch.tensor([timestep], device=z_t.device),
        encoder_attention_mask=mask,
        return_dict=False,
    )[0]
    return out.float()


def capture(tf, layers: list[int], grid: tuple[int, int, int]):
    """Hook `transformer_blocks[i]`; LTX blocks return flat video tokens only."""
    import contextlib

    f, hh, ww = grid
    store = {}
    handles = []

    def make_hook(idx):
        def hook(_m, _i, output):
            h = output[0] if isinstance(output, tuple) else output
            b, seq, d = h.shape
            assert seq == f * hh * ww, f"seq {seq} != {f}*{hh}*{ww}"
            store[idx] = h.float().view(b, f, hh * ww, d).mean(dim=2)[0].detach().cpu()
        return hook

    @contextlib.contextmanager
    def ctx():
        try:
            for i in layers:
                handles.append(tf.transformer_blocks[i].register_forward_hook(make_hook(i)))
            yield store
        finally:
            for h in handles:
                h.remove()

    return ctx()


def pooled_raw_pixel(px: torch.Tensor, f_lat: int, size: int = 32) -> torch.Tensor:
    """Per-latent-frame 32x32 RGB, grouped to match LTX's 8x temporal compression.

    Uses `f_lat` so the pixel bound covers the same span of the clip as the DiT features;
    the other families use a 4x grouping and would silently mismatch here.
    """
    x = (px[0].permute(1, 0, 2, 3) + 1.0) * 0.5
    x = torch.nn.functional.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    n = x.shape[0]
    idx = torch.linspace(0, n - 1, f_lat).round().long()
    groups = []
    edges = torch.linspace(-0.5, n - 0.5, f_lat + 1)
    for i in range(f_lat):
        lo, hi = int(round(float(edges[i]) + 0.5)), int(round(float(edges[i + 1]) + 0.5))
        hi = max(hi, lo + 1)
        groups.append(x[lo:hi].mean(dim=0))
    del idx
    return torch.stack(groups, dim=0).flatten(1).half().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(MANIFESTS), required=True)
    ap.add_argument("--weights", choices=["real", "random"], default="real")
    ap.add_argument("--variant", choices=["plain", "cj"], default="plain")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--t-grid", type=int, nargs="+", default=[600])
    args = ap.parse_args()

    if args.dataset == "likephys" and args.variant == "cj":
        raise SystemExit("LikePhys has its own color_change clips")

    dev = "cuda:0"
    pre = W.preflight(dev)
    man = json.loads(MANIFESTS[args.dataset].read_text())
    clips = man if args.variant == "plain" else [c for c in man if c["label"] == 0]
    clips = [c for i, c in enumerate(clips) if i % args.num_shards == args.shard]
    if args.limit:
        clips = clips[: args.limit]

    suffix = "" if args.variant == "plain" else f"_{args.variant}"
    od = ROOT / "cache" / f"mochi_feats_{args.weights}_{args.dataset}{suffix}"
    od.mkdir(parents=True, exist_ok=True)
    print(f"mochi/{args.weights}/{args.dataset}/{args.variant} shard {args.shard}/"
          f"{args.num_shards}: {len(clips)} clips -> {od}", flush=True)
    print("preflight:", json.dumps(pre), flush=True)

    from diffusers import AutoencoderKLMochi, MochiTransformer3DModel

    snap = snapshot()
    vae = AutoencoderKLMochi.from_pretrained(str(snap / "vae"), torch_dtype=torch.float32).to(dev).eval()
    src = str(snap / "transformer")
    if args.weights == "random":
        cfg = MochiTransformer3DModel.load_config(src)
        with torch.device("meta"):
            tf = MochiTransformer3DModel.from_config(cfg)
        tf = tf.to_empty(device=dev)
        _init_norm_aware(tf)
        tf = tf.to(torch.bfloat16).eval()
    else:
        tf = MochiTransformer3DModel.from_pretrained(src, torch_dtype=torch.bfloat16).to(dev).eval()
    layers = list(range(tf.config.num_layers))
    pe, pe_blob = W.load_prompt_embeds(str(OUT / "prompt_embeds_empty_cogvideox.pt"), dev,
                                       torch.bfloat16)
    global _SEQ_LENS
    _SEQ_LENS = pe_blob["seq_lens"].to(torch.int64).reshape(-1)
    print(f"prompt embeds {tuple(pe.shape)}, seq_lens {_SEQ_LENS.tolist()}", flush=True)
    env = W.env_stamp()

    t0 = time.time()
    for i, clip in enumerate(clips):
        cid = clip.get("clip_id", "")
        dst = od / (cid.replace("/", "__") + ".pt")
        if dst.exists():
            continue
        try:
            frames = C.read_spanning(clip["path"], N_FRAMES, SIZE)
        except Exception as exc:
            print(f"SKIP {cid}: {exc}", flush=True)
            continue

        px = W.preprocess_frames(frames, SIZE, SIZE, dev)
        if args.variant == "cj":
            px = colour_jitter(px)
        z0 = encode(vae, px)
        grid = latent_grid(z0)

        rec = {
            "clip": clip, "env": env, "preflight": pre,
            "backbone": args.weights, "variant": args.variant, "dataset": args.dataset,
            "grid": list(grid),
            "config": {"height": SIZE, "width": SIZE, "frames": N_FRAMES,
                       "sampling": "uniform over whole clip",
                       "t_grid": args.t_grid, "feat_seeds": FEAT_SEEDS,
                       "model": REPO, "revision": REVISION, "n_layers": len(layers),
                       "family": "mochi", "latent_channels": int(z0.shape[2]),
                       "colour_jitter": ({"hue": 0.35, "sat": 1.4}
                                         if args.variant == "cj" else None),
                       "tag": f"e8_mochi_{args.dataset}_{args.variant}"},
            "vae_latent_pooled": z0[0].mean(dim=(2, 3)).half().cpu(),
            "raw_pixel_32": pooled_raw_pixel(px, grid[0]),
            "feats": {},
        }
        for t in args.t_grid:
            rec["feats"][str(t)] = {}
            for seed in FEAT_SEEDS:
                g = torch.Generator(device=z0.device).manual_seed(int(seed))
                eps = torch.randn(z0.shape, generator=g, device=z0.device, dtype=torch.float32)
                sigma = float(t) / 1000.0
                z_t = ((1.0 - sigma) * z0 + sigma * eps).to(z0.dtype)
                with capture(tf, layers, grid) as hs:
                    forward_v(tf, z_t, t, pe, grid)
                rec["feats"][str(t)][str(seed)] = torch.stack(
                    [hs[k] for k in layers], dim=0).half()

        torch.save(rec, dst)
        if (i + 1) % 25 == 0 or i == 0:
            el = time.time() - t0
            print(f"[{i+1}/{len(clips)}] {el:.0f}s ({el/(i+1):.2f}s/clip) "
                  f"peak={torch.cuda.max_memory_allocated()/2**30:.1f}GB grid={grid}", flush=True)

    print(f"shard {args.shard} done in {time.time()-t0:.0f}s", flush=True)


@torch.no_grad()
def _init_norm_aware(model, seed: int = 0) -> None:
    """Same rule as CogVideoX's random control: weights normal, norm weights 1, biases 0.

    LTX is `use_rotary=True` and builds position information from `self.rope`, which has no
    parameters, so no positional buffer needs special treatment here.
    """
    import numpy as np

    g = torch.Generator(device="cpu").manual_seed(seed)
    for name, p in model.named_parameters():
        if p.ndim >= 2:
            fan_in = int(np.prod(p.shape[1:]))
            std = (2.0 / max(fan_in, 1)) ** 0.5
            p.copy_(torch.empty(p.shape, dtype=torch.float32).normal_(0.0, std, generator=g).to(p.dtype))
        elif name.endswith("norm.weight") or name.endswith("norm1.weight") or name.endswith("norm2.weight"):
            p.fill_(1.0)
        else:
            p.zero_()
    for _n, b in model.named_buffers():
        if b.is_floating_point():
            b.zero_()


if __name__ == "__main__":
    main()
