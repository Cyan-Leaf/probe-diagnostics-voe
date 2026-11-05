"""E14 extraction: HunyuanVideo (12.8B) under the E8 protocol.

Why a separate script: three things differ from every other backbone here, and two of them
fail *loudly* rather than silently — which is worth stating, because this project already lost
a full day to a silent one (the CogVideoX-5b RoPE bug, E8 §12).

  1. **Latent layout is [B,C,F,H,W]**, not frame-major. Conv3d raises on the wrong order, so
     this one is safe to get wrong once.
  2. **Two block stacks.** 20 double-stream (`transformer_blocks`, video+text joint) followed
     by 40 single-stream (`single_transformer_blocks`, video only). The reference cell L16
     lands in the *double* stack and the 53.3%-depth cell L32 lands in the *single* stack, so
     those two cells are not two depths of the same quantity — E14_PREREG.md §3 says so and
     the report must repeat it.
  3. **Two text conditions.** LLaMA-3-8B hidden states (1000x4096) AND CLIP pooled (768).
     Both are required by `forward`; passing neither fails, passing the wrong one silently
     conditions the model on nonsense. They are precomputed once into
     `prompt_embeds_hunyuan.pt`.

Only `transformer_blocks` and `single_transformer_blocks` are hooked. `context_embedder`
(which holds the token refiner) runs before either stack and is *not* reported: it sees text
only, so a probe on it would measure the prompt, not the video.

Run (one process per GPU):
  HF_HOME=... CUDA_VISIBLE_DEVICES=i python scripts/e14_extract.py \
      --dataset intphys2 --weights real --variant plain
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cogvideox_probe_lib as C
import wan_probe_lib as W
from e8_extract import colour_jitter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E14_hunyuan"
FEAT_SEEDS = [0, 1, 2]

REPO = "hunyuanvideo-community/HunyuanVideo"
REVISION = "e8c2aaa66fe3742a32c11a6766aecbf07c56e773"
N_FRAMES = 49
SIZE = 512
T_GRID = [600]        # flow matching, same convention as Wan/LTX/Mochi; no SNR-matched cell
PATCH = 2             # transformer patch_size (spatial)
PATCH_T = 1           # transformer patch_size_t (temporal)

MANIFESTS = {
    "intphys2": ROOT / "results" / "E7_intphys2" / "manifest.json",
    "likephys": ROOT / "results" / "T3_null_floor" / "manifest.json",
}


def snapshot() -> Path:
    import os

    d = (Path(os.environ["HF_HOME"]) / "hub"
         / ("models--" + REPO.replace("/", "--")) / "snapshots" / REVISION)
    if not d.exists():
        raise FileNotFoundError(f"{d} missing — download {REPO} first")
    return d


def out_dir(weights: str, dataset: str, variant: str) -> Path:
    suffix = "" if variant == "plain" else f"_{variant}"
    return ROOT / "cache" / f"hunyuan_feats_{weights}_{dataset}{suffix}"


@torch.no_grad()
def encode(vae, px: torch.Tensor) -> torch.Tensor:
    """[1,3,F,H,W] in [-1,1] -> [1,C,F',H',W'] (HunyuanVideo's own layout, NOT frame-major)."""
    z = vae.encode(px.to(next(vae.parameters()).dtype)).latent_dist.mode()
    return (z.float() * float(vae.config.scaling_factor))


def latent_grid(z: torch.Tensor) -> tuple[int, int, int]:
    """Grid the TRANSFORMER sees: the VAE grid divided by the patch size.

    The VAE gives 13x64x64 for 512x512x49; `patch_size=2, patch_size_t=1` folds that to
    13x32x32 = 13312 tokens, the same budget as Wan and CogVideoX. Returning the VAE grid
    would make the hook's `seq == f*h*h` assertion fail on the true token count.
    """
    _, _, f, h, w = z.shape
    return f // PATCH_T, h // PATCH, w // PATCH


@torch.no_grad()
def forward_v(tf, z_t, timestep, pe, mask, pooled, dev) -> None:
    tf(hidden_states=z_t.to(tf.dtype), timestep=torch.tensor([float(timestep)], device=dev),
       encoder_hidden_states=pe, encoder_attention_mask=mask,
       pooled_projections=pooled, guidance=torch.tensor([1.0], device=dev),
       return_dict=False)


def capture(tf, layers: list[int], grid: tuple[int, int, int]):
    """Hook the 60 blocks as one indexed list: 0-19 double, 20-59 single.

    Both return [B, seq, D] of video tokens only (text lives inside the double blocks' joint
    attention and is not part of the output), so one hook works for both stacks.
    """
    import contextlib

    f, hh, ww = grid
    blocks = list(tf.transformer_blocks) + list(tf.single_transformer_blocks)
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
                handles.append(blocks[i].register_forward_hook(make_hook(i)))
            yield store
        finally:
            for h in handles:
                h.remove()

    return ctx()


def pooled_raw_pixel(px: torch.Tensor, f_lat: int, size: int = 32) -> torch.Tensor:
    """Per-latent-frame 32x32 RGB. Hunyuan's VAE compresses 4x temporally, like Wan/CogVideoX."""
    x = (px[0].permute(1, 0, 2, 3) + 1.0) * 0.5
    x = torch.nn.functional.interpolate(x, size=(size, size), mode="bilinear",
                                       align_corners=False)
    n = x.shape[0]
    groups = [x[0:1]] + [x[1 + 4 * i: 1 + 4 * (i + 1)] for i in range((n - 1) // 4)]
    return torch.stack([g.mean(dim=0) for g in groups], dim=0).flatten(1).half().cpu()


def _init_norm_aware(model, seed: int = 0) -> None:
    g = torch.Generator(device="cpu").manual_seed(seed)
    for name, p in model.named_parameters():
        p.requires_grad_(False)
        if p.ndim >= 2:
            fan_in = int(np.prod(p.shape[1:]))
            std = (2.0 / max(fan_in, 1)) ** 0.5
            p.copy_(torch.empty(p.shape, dtype=torch.float32)
                    .normal_(0.0, std, generator=g).to(p.dtype))
        elif "norm" in name and name.endswith("weight"):
            p.fill_(1.0)
        else:
            p.zero_()
    for _n, b in model.named_buffers():
        if b.is_floating_point():
            b.zero_()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(MANIFESTS), default="intphys2")
    ap.add_argument("--weights", choices=["real", "random"], default="real")
    ap.add_argument("--variant", choices=["plain", "cj"], default="plain")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if args.dataset == "likephys" and args.variant == "cj":
        raise SystemExit("LikePhys carries its own color_change clips; no jitter variant needed")

    dev = "cuda:0"
    pre = W.preflight(dev)

    man = json.loads(MANIFESTS[args.dataset].read_text())
    clips = man if args.variant == "plain" else [c for c in man if c["label"] == 0]
    clips = [c for i, c in enumerate(clips) if i % args.num_shards == args.shard]
    if args.limit:
        clips = clips[: args.limit]

    od = out_dir(args.weights, args.dataset, args.variant)
    od.mkdir(parents=True, exist_ok=True)
    print(f"hunyuan/{args.weights}/{args.dataset}/{args.variant} shard "
          f"{args.shard}/{args.num_shards}: {len(clips)} clips -> {od}", flush=True)
    print("preflight:", json.dumps(pre), flush=True)

    from diffusers import AutoencoderKLHunyuanVideo, HunyuanVideoTransformer3DModel

    snap = snapshot()
    vae = AutoencoderKLHunyuanVideo.from_pretrained(
        str(snap / "vae"), torch_dtype=torch.float32).to(dev).eval()
    src = str(snap / "transformer")
    if args.weights == "random":
        cfg = HunyuanVideoTransformer3DModel.load_config(src)
        with torch.device("meta"):
            tf = HunyuanVideoTransformer3DModel.from_config(cfg)
        tf = tf.to_empty(device=dev)
        _init_norm_aware(tf)
        tf = tf.to(torch.bfloat16).eval()
    else:
        tf = HunyuanVideoTransformer3DModel.from_pretrained(
            src, torch_dtype=torch.bfloat16).to(dev).eval()

    n_blocks = len(tf.transformer_blocks) + len(tf.single_transformer_blocks)
    layers = list(range(n_blocks))
    blob = torch.load(OUT / "prompt_embeds_hunyuan.pt", map_location="cpu")
    pe = blob["llama_embeds"].to(dev, torch.bfloat16)
    pooled = blob["clip_pooled"].to(dev, torch.bfloat16)
    mask = torch.ones(1, pe.shape[1], device=dev, dtype=torch.bfloat16)
    env = W.env_stamp()

    t0 = time.time()
    for i, clip in enumerate(clips):
        cid = clip.get("clip_id", "")
        dst = od / (cid.replace("/", "__") + ".pt")
        if dst.exists():
            continue
        try:
            # Spanning, not first-N: the violation is unannotated in IntPhys 2, and the first
            # 49 native frames cover ~8% of a 636-frame clip (same rule as every other
            # extractor here). `wan_probe_lib.read_video_frames` defaults to first-N with a
            # stride, so it is NOT used.
            frames = C.read_spanning(clip["path"], N_FRAMES, SIZE)
        except Exception as exc:
            print(f"SKIP {cid}: {exc}", flush=True)
            continue

        px = W.preprocess_frames(frames, SIZE, SIZE, dev)
        if args.variant == "cj":
            px = colour_jitter(px)
        z0 = encode(vae, px)
        grid = latent_grid(z0)
        assert grid == (13, 32, 32), f"unexpected token grid {grid} (expected 13x32x32)"

        rec = {
            "clip": clip, "env": env, "preflight": pre,
            "backbone": args.weights, "variant": args.variant, "dataset": args.dataset,
            "grid": list(grid),
            "config": {"height": SIZE, "width": SIZE, "frames": N_FRAMES,
                       "sampling": "uniform over whole clip",
                       "t_grid": T_GRID, "feat_seeds": FEAT_SEEDS,
                       "model": REPO, "revision": REVISION, "n_layers": n_blocks,
                       "n_double": len(tf.transformer_blocks),
                       "n_single": len(tf.single_transformer_blocks),
                       "family": "hunyuan", "latent_channels": int(z0.shape[1]),
                       "latent_layout": "B,C,F,H,W",
                       "colour_jitter": ({"hue": 0.35, "sat": 1.4}
                                         if args.variant == "cj" else None),
                       "tag": f"e14_hunyuan_{args.dataset}_{args.variant}"},
            "vae_latent_pooled": z0[0].mean(dim=(2, 3)).transpose(0, 1).half().cpu(),
            "raw_pixel_32": pooled_raw_pixel(px, grid[0]),
            "feats": {},
        }

        for t in T_GRID:
            rec["feats"][str(t)] = {}
            for seed in FEAT_SEEDS:
                sigma = float(t) / 1000.0
                g = torch.Generator(device=z0.device).manual_seed(int(seed))
                eps = torch.randn(z0.shape, generator=g, device=z0.device,
                                  dtype=torch.float32)
                z_t = ((1.0 - sigma) * z0 + sigma * eps).to(z0.dtype)
                with capture(tf, layers, grid) as hs:
                    forward_v(tf, z_t, t, pe, mask, pooled, dev)
                rec["feats"][str(t)][str(seed)] = torch.stack(
                    [hs[k] for k in layers], dim=0).half()

        torch.save(rec, dst)
        if (i + 1) % 25 == 0 or i == 0:
            el = time.time() - t0
            print(f"[{i+1}/{len(clips)}] {el:.0f}s ({el/(i+1):.2f}s/clip) grid={grid} "
                  f"peak={torch.cuda.max_memory_allocated()/2**30:.1f}GB", flush=True)

    print(f"shard {args.shard} done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
