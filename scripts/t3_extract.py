"""T3/T4 extraction pass: cache denoising errors (multi-seed) and pooled DiT states.

Protocol is fixed in results/T3_null_floor/protocol_as_executed.md.

One output .pt per clip per backbone:
  {
    "clip": {...manifest entry...},
    "errors": {t: {seed: {"mse": float, "per_frame": [13]}}},
    "feats":  {t: {seed: fp16 tensor [n_layers, 13, 3072]}},
    "vae_latent_pooled": fp16 [13, 48],
    "raw_pixel_32": fp16 [13, 3072],
    "env": {...}, "preflight": {...},
  }

Run (one process per healthy GPU):
  export HF_HOME=... ; CUDA_VISIBLE_DEVICES=1 uv run --no-project python scripts/t3_extract.py \
      --shard 0 --num-shards 3 --backbone real
"""

import argparse
import json
import time
from pathlib import Path

import torch

import wan_probe_lib as W

ROOT = Path(__file__).resolve().parents[1]
T_GRID = [200, 400, 600, 800, 950]
SEEDS_ERR = [0, 1, 2, 3, 4]
FEAT_SEEDS = {600: [0, 1, 2]}  # other timesteps: seed 0 only
DEFAULT_SEED = 0


def pooled_raw_pixel(frames, size: int = 32) -> torch.Tensor:
    """[F,H,W,3] uint8 -> per-latent-frame 32x32 RGB flattened, matching the latent frame count.

    Wan's causal VAE maps frame 0 to latent frame 0 and each subsequent group of 4 frames to one
    latent frame, so we average the pixels of the corresponding source frames.
    """
    x = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
    x = torch.nn.functional.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    groups = [x[0:1]] + [x[1 + 4 * i : 1 + 4 * (i + 1)] for i in range((x.shape[0] - 1) // 4)]
    per_latent = torch.stack([g.mean(dim=0) for g in groups], dim=0)  # [F', 3, s, s]
    return per_latent.flatten(1).half()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--backbone", choices=["real", "random"], default="real")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--frames", type=int, default=49)
    ap.add_argument("--manifest", default=str(ROOT / "results/T3_null_floor/manifest.json"))
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--embeds", default=str(ROOT / "results/T2_env/prompt_embeds_empty.pt"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--t-grid", type=int, nargs="+", default=None,
                    help="override T_GRID (default keeps legacy: real=all, random=[600])")
    ap.add_argument("--feat-seeds", type=str, default=None,
                    help="comma-separated t:s0,s1,... pairs (default: t=600 -> 0,1,2; else -> 0)")
    ap.add_argument("--err-seeds", type=int, nargs="+", default=None,
                    help="override SEEDS_ERR (default: real=[0..4], random=[0])")
    ap.add_argument("--tag", default="", help="free label stored in config for this run")
    args = ap.parse_args()

    dev = "cuda:0"
    pre = W.preflight(dev)
    print("preflight:", json.dumps(pre), flush=True)

    outdir = Path(args.outdir or ROOT / "cache" / f"feats_{args.backbone}")
    outdir.mkdir(parents=True, exist_ok=True)

    clips = json.loads(Path(args.manifest).read_text())
    clips = [c for i, c in enumerate(clips) if i % args.num_shards == args.shard]
    if args.limit:
        clips = clips[: args.limit]
    print(f"shard {args.shard}/{args.num_shards}: {len(clips)} clips -> {outdir}", flush=True)

    vae = W.load_vae(dev, torch.float32)
    tf = W.load_transformer(dev, torch.bfloat16, random_init=(args.backbone == "random"))
    layers = list(range(tf.config.num_layers))
    pe, _ = W.load_prompt_embeds(args.embeds, dev, torch.bfloat16)
    env = W.env_stamp()

    if args.t_grid is not None:
        t_grid = args.t_grid
    else:
        t_grid = T_GRID if args.backbone == "real" else [600]

    if args.err_seeds is not None:
        seeds_err = args.err_seeds
    else:
        seeds_err = SEEDS_ERR if args.backbone == "real" else [0]

    if args.feat_seeds is not None:
        feat_seeds_override = {}
        for pair in args.feat_seeds.split():
            t_str, s_str = pair.split(":")
            feat_seeds_override[int(t_str)] = [int(s) for s in s_str.split(",")]
    else:
        feat_seeds_override = None

    # Record what was *actually* extracted, not the module-level default: otherwise a
    # grid-completion run would stamp the legacy schedule into its own provenance block.
    effective_feat_seeds = {
        t: (feat_seeds_override.get(t, [DEFAULT_SEED]) if feat_seeds_override is not None
            else FEAT_SEEDS.get(t, [DEFAULT_SEED]))
        for t in t_grid
    }
    print(f"t_grid={t_grid} err_seeds={seeds_err} feat_seeds={effective_feat_seeds}", flush=True)

    t_start = time.time()
    for i, clip in enumerate(clips):
        dst = outdir / (clip["clip_id"].replace("/", "__") + ".pt")
        if dst.exists():
            continue
        try:
            frames = W.read_video_frames(clip["path"], args.frames)
        except ValueError as exc:
            print(f"SKIP {clip['clip_id']}: {exc}", flush=True)
            continue
        px = W.preprocess_frames(frames, args.height, args.width, dev)
        z0 = W.encode_latents(vae, px, normalize=True)
        grid = W.latent_grid(z0)

        rec = {
            "clip": clip,
            "env": env,
            "preflight": pre,
            "backbone": args.backbone,
            "grid": list(grid),
            "config": {"height": args.height, "width": args.width, "frames": args.frames,
                       "t_grid": t_grid, "seeds_err": seeds_err,
                       "feat_seeds": effective_feat_seeds,
                       "tag": args.tag, "n_layers": len(layers)},
            "vae_latent_pooled": z0[0].mean(dim=(2, 3)).transpose(0, 1).half().cpu(),  # [F', 48]
            "raw_pixel_32": pooled_raw_pixel(frames),
            "errors": {},
            "feats": {},
        }

        for t_int in t_grid:
            rec["errors"][str(t_int)] = {}
            feat_seeds = effective_feat_seeds[t_int]
            for seed in sorted(set(seeds_err) | set(feat_seeds)):
                z_t, _, v_tgt = W.flow_noise(z0, t_int, seed=seed)
                ts = W.per_token_timestep(z_t, t_int)
                want_feats = seed in feat_seeds
                if want_feats:
                    with W.capture_blocks(tf, layers, grid) as hs:
                        v_pred = W.forward_v(tf, z_t, ts, pe)
                    stack = torch.stack([hs.pooled[k] for k in layers], dim=0).half()
                    rec["feats"].setdefault(str(t_int), {})[str(seed)] = stack
                else:
                    v_pred = W.forward_v(tf, z_t, ts, pe)
                if seed in seeds_err:
                    e = W.denoising_error(v_pred, v_tgt)
                    rec["errors"][str(t_int)][str(seed)] = {
                        "mse": e["mse"],
                        "per_frame": [round(x, 6) for x in e["per_frame"]],
                    }

        torch.save(rec, dst)
        if (i + 1) % 10 == 0 or i == 0:
            el = time.time() - t_start
            print(f"[{i+1}/{len(clips)}] {clip['clip_id']} {el:.0f}s "
                  f"({el/(i+1):.2f}s/clip) peak={torch.cuda.max_memory_allocated()/2**30:.1f}GB",
                  flush=True)

    print(f"shard {args.shard} done in {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
