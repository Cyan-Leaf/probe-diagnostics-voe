"""E8 extraction, any registered backbone (see scripts/e8_backbones.py).

Same record layout as `e8_extract.py`, produced through the family adapter instead of the
CogVideoX-specific helper.  Added so the size ladder (Wan 1.3B / 14B) and the second
CogVideoX size reuse the E8 protocol verbatim rather than a lookalike reimplementation:
the analysis scripts read both, and a difference between two backbones can then only come
from the model.

Record is one .pt per clip per (backbone, weights, dataset, variant), holding every
pre-registered timestep and all 30+ layers, so a later probe sweep costs no GPU time.

Run (one process per GPU):
  HF_HOME=... CUDA_VISIBLE_DEVICES=i python scripts/e8_extract_generic.py \
      --spec wan21_1b3 --dataset intphys2 --weights real --variant plain \
      --shard i --num-shards 4
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e8_backbones as B
import wan_probe_lib as W

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E8_second_backbone"
N_FRAMES = 49
SIZE = 512
FEAT_SEEDS = [0, 1, 2]

MANIFESTS = {
    "intphys2": ROOT / "results" / "E7_intphys2" / "manifest.json",
    "likephys": ROOT / "results" / "T3_null_floor" / "manifest.json",
}

HUE_SHIFT = 0.35
SAT_SCALE = 1.4


def colour_jitter(px: torch.Tensor, hue: float = HUE_SHIFT, sat: float = SAT_SCALE) -> torch.Tensor:
    """Identical to e7_e8's control: one fixed HSV shift per clip, same on every frame."""
    x = (px[0].permute(1, 0, 2, 3) + 1.0) * 0.5
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    mx, _ = x.max(dim=1)
    mn, _ = x.min(dim=1)
    d = mx - mn
    eps = 1e-8

    h = torch.zeros_like(mx)
    m = (mx == r) & (d > eps)
    h[m] = (((g - b)[m] / d[m]) % 6.0) / 6.0
    m = (mx == g) & (d > eps)
    h[m] = (((b - r)[m] / d[m]) + 2.0) / 6.0
    m = (mx == b) & (d > eps)
    h[m] = (((r - g)[m] / d[m]) + 4.0) / 6.0

    s = torch.where(mx > eps, d / (mx + eps), torch.zeros_like(mx))
    v = mx
    h = (h + hue) % 1.0
    s = (s * sat).clamp(0.0, 1.0)

    i = (h * 6.0).floor()
    f = h * 6.0 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    i = i.remainder(6).long()
    out = torch.stack([
        torch.where(i == 0, v, torch.where(i == 1, q, torch.where(
            i == 2, p, torch.where(i == 3, p, torch.where(i == 4, t, v))))),
        torch.where(i == 0, t, torch.where(i == 1, v, torch.where(
            i == 2, v, torch.where(i == 3, q, torch.where(i == 4, p, p))))),
        torch.where(i == 0, p, torch.where(i == 1, p, torch.where(
            i == 2, t, torch.where(i == 3, v, torch.where(i == 4, v, q))))),
    ], dim=1)
    return (out * 2.0 - 1.0).permute(1, 0, 2, 3).unsqueeze(0).contiguous()


def read_spanning(path, n: int, size: int) -> "object":
    import numpy as np
    import decord

    vr = decord.VideoReader(str(path), width=size, height=size)
    idx = np.linspace(0, len(vr) - 1, n).round().astype(int).tolist()
    return vr.get_batch(idx).asnumpy()


def out_dir(spec: B.Spec, weights: str, dataset: str, variant: str) -> Path:
    suffix = "" if variant == "plain" else f"_{variant}"
    return ROOT / "cache" / f"{spec.key}_feats_{weights}_{dataset}{suffix}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, choices=list(B.REGISTRY))
    ap.add_argument("--dataset", choices=list(MANIFESTS), required=True)
    ap.add_argument("--weights", choices=["real", "random"], default="real")
    ap.add_argument("--variant", choices=["plain", "cj"], default="plain")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--t-grid", type=int, nargs="+", default=None)
    args = ap.parse_args()

    spec = B.REGISTRY[args.spec]
    if args.dataset == "likephys" and args.variant == "cj":
        raise SystemExit("LikePhys has its own color_change clips; no jitter variant needed")

    dev = "cuda:0"
    pre = W.preflight(dev)

    man = json.loads(MANIFESTS[args.dataset].read_text())
    clips = man if args.variant == "plain" else [c for c in man if c["label"] == 0]
    clips = [c for i, c in enumerate(clips) if i % args.num_shards == args.shard]
    if args.limit:
        clips = clips[: args.limit]

    # For a flow-matching model the SNR-matched timestep IS 600, so dedupe — otherwise the
    # grid is [600, 600] and every feature is extracted twice under the same key.
    t_grid = args.t_grid or list(dict.fromkeys([600, B.snr_matched_t(spec)]))

    od = out_dir(spec, args.weights, args.dataset, args.variant)
    od.mkdir(parents=True, exist_ok=True)
    print(f"{spec.key}/{args.weights}/{args.dataset}/{args.variant} shard "
          f"{args.shard}/{args.num_shards}: {len(clips)} clips -> {od}", flush=True)
    print("preflight:", json.dumps(pre), flush=True)
    print(f"t grid {t_grid} (family={spec.family}, schedule={spec.schedule})", flush=True)

    vae = B.load_vae(spec, dev, torch.float32)
    tf = B.load_transformer(spec, dev, torch.bfloat16, random_init=(args.weights == "random"))
    layers = list(range(spec.n_layers))
    pe, _ = W.load_prompt_embeds(B.text_embeds_path(spec), dev, torch.bfloat16)
    env = W.env_stamp()

    t0 = time.time()
    for i, clip in enumerate(clips):
        cid = clip.get("clip_id", "")
        dst = od / (cid.replace("/", "__") + ".pt")
        if dst.exists():
            continue
        try:
            frames = read_spanning(clip["path"], N_FRAMES, SIZE)
        except Exception as exc:
            print(f"SKIP {cid}: {exc}", flush=True)
            continue

        px = W.preprocess_frames(frames, SIZE, SIZE, dev)
        if args.variant == "cj":
            px = colour_jitter(px)
        z0 = B.encode_latents(spec, vae, px)
        grid = B.latent_grid(spec, z0)

        # raw-pixel pooling must follow the model's own temporal compression, otherwise the
        # pixel bound silently covers a different span of the clip than the DiT features.
        rec = {
            "clip": clip, "env": env, "preflight": pre,
            "backbone": args.weights, "variant": args.variant, "dataset": args.dataset,
            "grid": list(grid),
            "config": {"height": SIZE, "width": SIZE, "frames": N_FRAMES,
                       "sampling": "uniform over whole clip", "t_grid": t_grid,
                       "feat_seeds": FEAT_SEEDS, "model": spec.repo,
                       "revision": spec.revision, "n_layers": len(layers),
                       "family": spec.family, "latent_channels": spec.latent_channels,
                       "colour_jitter": ({"hue": HUE_SHIFT, "sat": SAT_SCALE}
                                         if args.variant == "cj" else None),
                       "tag": f"e8_{spec.key}_{args.dataset}_{args.variant}"},
            "vae_latent_pooled": C_pool(z0, spec),
            "raw_pixel_32": raw_pixel(px),
            "feats": {},
        }

        for t in t_grid:
            rec["feats"][str(t)] = {}
            for seed in FEAT_SEEDS:
                z_t, _v = B.noise(spec, z0, t, seed=seed)
                with B.capture(spec, tf, layers, grid) as hs:
                    B.forward(spec, tf, z_t, pe, t, grid)
                rec["feats"][str(t)][str(seed)] = torch.stack(
                    [hs.pooled[k] for k in layers], dim=0).half()

        torch.save(rec, dst)
        if (i + 1) % 25 == 0 or i == 0:
            el = time.time() - t0
            print(f"[{i+1}/{len(clips)}] {el:.0f}s ({el/(i+1):.2f}s/clip) "
                  f"peak={torch.cuda.max_memory_allocated()/2**30:.1f}GB grid={grid}",
                  flush=True)

    print(f"shard {args.shard} done in {time.time()-t0:.0f}s", flush=True)


def C_pool(z0, spec):
    """Per-latent-frame pooled VAE latent, in a frame-major layout for both families."""
    if spec.family == "wan":
        return z0[0].mean(dim=(2, 3)).transpose(0, 1).half().cpu()
    return z0[0].mean(dim=(2, 3)).half().cpu()


def raw_pixel(px: torch.Tensor, size: int = 32) -> torch.Tensor:
    """Per-latent-frame 32x32 RGB. Grouping assumes 4x temporal compression, which holds for
    both Wan's causal VAE and CogVideoX's; the grid is asserted against the latent shape."""
    x = (px[0].permute(1, 0, 2, 3) + 1.0) * 0.5
    x = torch.nn.functional.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    groups = [x[0:1]] + [x[1 + 4 * i: 1 + 4 * (i + 1)] for i in range((x.shape[0] - 1) // 4)]
    return torch.stack([g.mean(dim=0) for g in groups], dim=0).flatten(1).half().cpu()


if __name__ == "__main__":
    main()
