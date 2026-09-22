"""S4 extraction: features for the 7 saturated scenarios under synthetic camera motion.

One process per GPU, sharded like t3_extract.py.  Writes to a NEW cache directory per
(level, backbone) so nothing existing is touched (hard rule 2).

Protocol pinned to match the S3 baseline exactly, so the level-0..3 AUCs are directly
comparable to the S3 number that selected these scenarios:
  t = 600, feature seed 0, all 30 layers, layer 16 used at analysis time.

Run (per level, per backbone):
  CUDA_VISIBLE_DEVICES=i python scripts/s4_extract.py --level 0 --backbone real \
      --shard i --num-shards 4
"""

import argparse
import json
import time
from pathlib import Path

import torch

import s4_camera as CAM
import wan_probe_lib as W

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ["ball_drop", "cloth_drape", "faucet", "fluid", "pendulum", "pyramid", "river"]
T_FIXED = 600
SEED_FIXED = 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, required=True, choices=sorted(CAM.LEVELS))
    ap.add_argument("--backbone", choices=["real", "random"], required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--frames", type=int, default=49)
    ap.add_argument("--manifest", default=str(ROOT / "results/T3_null_floor/manifest.json"))
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--embeds", default=str(ROOT / "results/T2_env/prompt_embeds_empty.pt"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda:0"
    pre = W.preflight(dev)
    print("preflight:", json.dumps(pre), flush=True)

    outdir = Path(args.outdir or ROOT / "cache" /
                  f"feats_{args.backbone}_cam{args.level}")
    outdir.mkdir(parents=True, exist_ok=True)

    clips = json.loads(Path(args.manifest).read_text())
    clips = [c for c in clips if c["scenario"] in SCENARIOS and c["group"] == "physics"]
    clips = [c for i, c in enumerate(clips) if i % args.num_shards == args.shard]
    if args.limit:
        clips = clips[: args.limit]
    print(f"level={args.level} backbone={args.backbone} shard {args.shard}/"
          f"{args.num_shards}: {len(clips)} clips -> {outdir}", flush=True)
    print(f"crop geometry: {json.dumps(CAM.crop_geometry(args.level))}", flush=True)

    vae = W.load_vae(dev, torch.float32)
    tf = W.load_transformer(dev, torch.bfloat16, random_init=(args.backbone == "random"))
    layers = list(range(tf.config.num_layers))
    pe, _ = W.load_prompt_embeds(args.embeds, dev, torch.bfloat16)
    env = W.env_stamp()

    # Single-entry cache, NOT a dict keyed by subgroup: one grid is
    # [49, 512, 512, 2] fp32 = 103 MB on device, and the 7x10 subgroups would pin 7.2 GB
    # and OOM the VAE.  The manifest is ordered so a subgroup's clips are consecutive, so
    # one slot gives the same hit rate.  Rebuilding is also safe -- build_grid is seeded
    # from (scenario, subgroup) and is bit-identical on repeat (selftest check 3).
    grid_key = None
    grid = realised = None
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

        key = (clip["scenario"], clip["subgroup"])
        if key != grid_key:
            grid = realised = None
            torch.cuda.empty_cache()
            grid, realised = CAM.build_grid(args.level, key[0], key[1], dev,
                                            frames=args.frames, size=args.height)
            grid_key = key

        px = CAM.apply_camera(px, grid)
        z0 = W.encode_latents(vae, px, normalize=True)
        gshape = W.latent_grid(z0)

        rec = {
            "clip": clip,
            "env": env,
            "preflight": pre,
            "backbone": args.backbone,
            "grid": list(gshape),
            "config": {"height": args.height, "width": args.width, "frames": args.frames,
                       "t_grid": [T_FIXED], "seeds_err": [SEED_FIXED],
                       "feat_seeds": {T_FIXED: [SEED_FIXED]},
                       "tag": f"s4_cam{args.level}", "n_layers": len(layers)},
            "camera": {"level": args.level,
                       "spec": CAM.LEVELS[args.level],
                       "realised": realised,
                       "crop": CAM.crop_geometry(args.level),
                       "trajectory_seed": CAM.trajectory_seed(*key),
                       "trajectory_key": f"{key[0]}/{key[1]}"},
            "vae_latent_pooled": z0[0].mean(dim=(2, 3)).transpose(0, 1).half().cpu(),
            "raw_pixel_32": CAM.pooled_raw_pixel_from_px(px),
            "errors": {},
            "feats": {},
        }

        z_t, _, v_tgt = W.flow_noise(z0, T_FIXED, seed=SEED_FIXED)
        ts = W.per_token_timestep(z_t, T_FIXED)
        with W.capture_blocks(tf, layers, gshape) as hs:
            v_pred = W.forward_v(tf, z_t, ts, pe)
        rec["feats"][str(T_FIXED)] = {
            str(SEED_FIXED): torch.stack([hs.pooled[k] for k in layers], dim=0).half()
        }
        e = W.denoising_error(v_pred, v_tgt)
        rec["errors"][str(T_FIXED)] = {
            str(SEED_FIXED): {"mse": e["mse"],
                              "per_frame": [round(x, 6) for x in e["per_frame"]]}
        }

        torch.save(rec, dst)
        if (i + 1) % 25 == 0 or i == 0:
            el = time.time() - t_start
            print(f"[{i+1}/{len(clips)}] {clip['clip_id']} {el:.0f}s "
                  f"({el/(i+1):.2f}s/clip) peak="
                  f"{torch.cuda.max_memory_allocated()/2**30:.1f}GB", flush=True)

    print(f"level={args.level} backbone={args.backbone} shard {args.shard} done in "
          f"{time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
