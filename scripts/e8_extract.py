"""E8 extraction: CogVideoX-2b features for the four lower bounds.

Produces one .pt per clip per (backbone, dataset, variant), mirroring E7's record layout so
the analysis can reuse the frozen probe unchanged.

Two timesteps are always written in the same record, per the pre-registration:

  t=600  literal  - matches PREREG.md's reference cell verbatim
  t=298  SNR-matched to Wan's t=600 (see cogvideox_probe_lib.noise_schedule)

Both are stored regardless of which one later looks better, so neither can be chosen after
seeing a result.

Run (one process per GPU):
  HF_HOME=... CUDA_VISIBLE_DEVICES=i python scripts/e8_extract.py \
      --dataset intphys2 --backbone real --variant plain --shard i --num-shards 4
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

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E8_second_backbone"
N_FRAMES = 49
SIZE = 512
FEAT_SEEDS = [0, 1, 2]
T_GRID = [C.T_LITERAL, C.T_SNR]

MANIFESTS = {
    "intphys2": ROOT / "results" / "E7_intphys2" / "manifest.json",
    "likephys": ROOT / "results" / "T3_null_floor" / "manifest.json",
}

# Reused verbatim from E7 so the appearance control is the *same* manipulation, not a
# lookalike: one fixed HSV shift per clip, identical on every frame, no temporal component.
HUE_SHIFT = 0.35
SAT_SCALE = 1.4


def colour_jitter(px: torch.Tensor, hue: float = HUE_SHIFT, sat: float = SAT_SCALE) -> torch.Tensor:
    """[1,3,F,H,W] in [-1,1] -> hue/saturation shifted identically on every frame.

    Physics-preserving by construction: fixed transform, no temporal component, no geometry
    change.  Copied from scripts/e7_extract.py rather than re-derived, so D2 compares the
    same perturbation across backbones.
    """
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


def out_dir(backbone: str, dataset: str, variant: str, tag: str = "") -> Path:
    """Records are skipped when they already exist, so a different t-grid has to go to a
    different directory — otherwise the whole sweep is silently skipped as "already done".
    """
    suffix = "" if variant == "plain" else f"_{variant}"
    tag_suffix = f"_{tag}" if tag else ""
    return ROOT / "cache" / f"cogvideox_feats_{backbone}_{dataset}{suffix}{tag_suffix}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(MANIFESTS), required=True)
    ap.add_argument("--backbone", choices=["real", "random"], default="real")
    ap.add_argument("--variant", choices=["plain", "cj"], default="plain")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--embeds", default=str(OUT / "prompt_embeds_empty_cogvideox.pt"))
    ap.add_argument("--t-grid", type=int, nargs="+", default=None,
                    help="override the two pre-registered timesteps (for the SNR sweep)")
    ap.add_argument("--tag", default="", help="free label stored in the record's config")
    args = ap.parse_args()

    if args.dataset == "likephys" and args.variant == "cj":
        raise SystemExit("LikePhys carries its own color_change clips; no jitter variant is needed "
                         "(see E8_PREREG.md §5)")

    dev = "cuda:0"
    pre = W.preflight(dev)

    man = json.loads(MANIFESTS[args.dataset].read_text())
    # The jitter control is defined on physics-legal clips only, matching E7.
    clips = man if args.variant == "plain" else [c for c in man if c["label"] == 0]
    clips = [c for i, c in enumerate(clips) if i % args.num_shards == args.shard]
    if args.limit:
        clips = clips[: args.limit]

    od = out_dir(args.backbone, args.dataset, args.variant, args.tag)
    od.mkdir(parents=True, exist_ok=True)
    print(f"{args.dataset}/{args.backbone}/{args.variant} shard {args.shard}/{args.num_shards}: "
          f"{len(clips)} clips -> {od}", flush=True)
    print("preflight:", json.dumps(pre), flush=True)

    sched = C.noise_schedule()
    t_grid = args.t_grid or T_GRID
    ab = sched["alphas_cumprod"]
    print("t grid " + ", ".join(
        f"{t}(sigma_eff={float((1-ab[t])**0.5):.4f},SNR={float(ab[t]/(1-ab[t])):.4f})"
        for t in t_grid), flush=True)

    vae = C.load_vae(dev, torch.float32)
    tf = C.load_transformer(dev, torch.bfloat16, random_init=(args.backbone == "random"))
    layers = list(range(tf.config.num_layers))
    pe, _ = C.load_text_embeds(args.embeds, dev, torch.bfloat16)
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

        px = C.preprocess_frames(frames, SIZE, SIZE, dev)
        if args.variant == "cj":
            px = colour_jitter(px)
        z0 = C.encode_latents(vae, px)
        grid = C.latent_grid(z0)
        assert grid == (13, 32, 32), f"unexpected token grid {grid} (expected 13x32x32)"

        rec = {
            "clip": clip,
            "env": env,
            "preflight": pre,
            "backbone": args.backbone,
            "variant": args.variant,
            "dataset": args.dataset,
            "grid": list(grid),
            "config": {
                "height": SIZE, "width": SIZE, "frames": N_FRAMES,
                "sampling": "uniform over whole clip",
                "t_grid": t_grid, "feat_seeds": FEAT_SEEDS,
                "sweep_tag": args.tag,
                "model": C.REPO, "revision": C.REVISION,
                "colour_jitter": ({"hue": HUE_SHIFT, "sat": SAT_SCALE}
                                  if args.variant == "cj" else None),
                "n_layers": len(layers), "tag": f"e8_cogvideox_{args.dataset}_{args.variant}" + (f"_{args.tag}" if args.tag else ""),
            },
            "vae_latent_pooled": z0[0].mean(dim=(2, 3)).half().cpu(),
            "raw_pixel_32": C.pooled_raw_pixel(px),
            "feats": {},
        }

        for t in t_grid:
            rec["feats"][str(t)] = {}
            for seed in FEAT_SEEDS:
                z_t, _, _ = C.flow_noise(z0, t, seed=seed)
                with C.capture_blocks(tf, layers, grid) as hs:
                    C.forward_v(tf, z_t, torch.tensor([t], device=dev), pe, grid=grid)
                rec["feats"][str(t)][str(seed)] = torch.stack(
                    [hs.pooled[k] for k in layers], dim=0).half()

        torch.save(rec, dst)
        if (i + 1) % 25 == 0 or i == 0:
            el = time.time() - t0
            print(f"[{i+1}/{len(clips)}] {el:.0f}s ({el/(i+1):.2f}s/clip) "
                  f"peak={torch.cuda.max_memory_allocated()/2**30:.1f}GB", flush=True)

    print(f"shard {args.shard} done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
