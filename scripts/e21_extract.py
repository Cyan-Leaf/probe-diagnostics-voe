"""E21: Physics-IQ real-world video — does a "physics" probe read physics, or read the camera?

Every dataset in this project is synthetically edited: IntPhys 2 and LikePhys both build an
"impossible" clip by modifying a rendered source. D2 tests whether the probe reads the edit,
and fails 34/34, but D2's control is a recolour WE applied. No experiment here has ever used
footage that was never edited.

Physics-IQ supplies 396 such clips (real cameras, real objects). Design frozen in
`results/E21_physics_iq_real/E21_PREREG.md`, written before extraction. The scope limit that
document establishes and this script enforces:

  take-1 / take-2 are TWO NATURAL RECORDINGS OF THE SAME SCENE. Both are physically legal.
  They are NOT a possible/impossible pair, so this is NOT a violation-detection task and must
  never be reported as one. What it measures is whether a physics-irrelevant, real-world
  contrast (two performances of one scene) is linearly readable from the same features.

Reading is identical to E7's IntPhys 2 path — 49 frames spanning the whole clip at 512px,
Wan2.1-T2V-1.3B, L16 / t=600, feature seeds 0/1/2 — so the numbers sit next to the E8 table.

Run (one process per GPU; 4 shards is comfortable):
  HF_HOME=... CUDA_VISIBLE_DEVICES=i python scripts/e21_extract.py --stage manifest
  HF_HOME=... CUDA_VISIBLE_DEVICES=i python scripts/e21_extract.py --stage extract \
      --backbone real --shard i --num-shards 4
"""

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wan_probe_lib as W
# Imported rather than re-derived: the raw-pixel lower bound must be the SAME statistic the
# IntPhys 2 / LikePhys tables use, otherwise the E21 bound cannot be compared to them.
from e7_extract import pooled_raw_pixel

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E21_physics_iq_real"
PIQ = ROOT / "data" / "physics_iq"
N_FRAMES = 49
SIZE = 512
T_FIXED = 600
FEAT_SEEDS = [0, 1, 2]

# `<idx>_full-videos_<fps>_perspective-<p>_take-<n>_trimmed-<scene>.mp4`
NAME_RX = re.compile(
    r"^(?P<idx>\d+)_full-videos_(?P<fps>\w+FPS)_perspective-(?P<persp>\w+?)_"
    r"take-(?P<take>\d)_trimmed-(?P<scene>.+)\.mp4$")


def build_manifest() -> list[dict]:
    """All 396 clips, keyed by (scene, perspective) so the takes pair up exactly.

    Every scene-perspective key carries both takes (198/198 verified in E21_PREREG §1); the
    assert below keeps that invariant from silently degrading if the data is ever re-fetched.
    """
    rows, keys = [], {}
    for take_dir in ("take-1", "take-2"):
        for p in sorted((PIQ / take_dir).glob("*.mp4")):
            m = NAME_RX.match(p.name)
            assert m, f"unparsable name: {p.name}"
            g = m.groupdict()
            key = f"{g['scene']}|{g['persp']}"
            keys.setdefault(key, set()).add(g["take"])
            rows.append({
                "clip_id": f"piq/{g['scene']}/{g['persp']}/take-{g['take']}",
                "path": str(p),
                "scene": g["scene"],
                "perspective": g["persp"],
                "take": int(g["take"]),
                "fps": g["fps"],
                "index": g["idx"],
                "key": key,
                # E21's two label axes. `perspective` is left/center/right; the positive
                # control in PREREG §4 uses left vs right only, so the binarisation is
                # recorded here as a flag rather than derived later.
                "label_take": int(g["take"]) - 1,
                "label_persp_lr": None,
                "bytes": p.stat().st_size,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            })
    assert all(v == {"1", "2"} for v in keys.values()), \
        f"unpaired keys: {[k for k, v in keys.items() if v != {'1','2'}][:5]}"
    lr = {"left": 0, "right": 1}
    for r in rows:
        r["label_persp_lr"] = lr.get(r["perspective"])
    return rows


def read_spanning(path: str, n: int) -> np.ndarray:
    """`n` frames spanning the entire clip. Same rule as E7: the event is not annotated, so
    the first n native frames would cover a fraction of the clip."""
    import decord

    vr = decord.VideoReader(str(path), width=SIZE, height=SIZE)
    total = len(vr)
    if total < n:
        raise ValueError(f"{path}: {total} frames < {n}")
    idx = np.linspace(0, total - 1, n).round().astype(int).tolist()
    return vr.get_batch(idx).asnumpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["manifest", "extract"], required=True)
    ap.add_argument("--backbone", choices=["real", "random"], default="real")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--embeds", default=str(ROOT / "results/T2_env/prompt_embeds_empty.pt"))
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    mpath = OUT / "manifest.json"

    if args.stage == "manifest":
        man = build_manifest()
        mpath.write_text(json.dumps(man, indent=1, ensure_ascii=False))
        print(f"{len(man)} clips, {len({c['scene'] for c in man})} scenes, "
              f"{len({c['key'] for c in man})} scene-perspective keys")
        for k in ("take", "perspective"):
            print(f"  {k:12s} {dict(Counter(c[k] for c in man))}")
        print(f"  label_take   {dict(Counter(c['label_take'] for c in man))}")
        print(f"  label_persp_lr {dict(Counter(c['label_persp_lr'] for c in man))}")
        print(f"wrote {mpath}")
        return

    man = json.loads(mpath.read_text())
    clips = [c for i, c in enumerate(man) if i % args.num_shards == args.shard]
    if args.limit:
        clips = clips[: args.limit]

    dev = "cuda:0"
    pre = W.preflight(dev)
    print("preflight:", json.dumps(pre), flush=True)

    outdir = Path(args.outdir or ROOT / "cache" / f"piq_feats_{args.backbone}")
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"physics_iq/{args.backbone} shard {args.shard}/{args.num_shards}: "
          f"{len(clips)} clips -> {outdir}", flush=True)

    vae = W.load_vae(dev, torch.float32)
    tf = W.load_transformer(dev, torch.bfloat16, random_init=(args.backbone == "random"))
    layers = list(range(tf.config.num_layers))
    pe, _ = W.load_prompt_embeds(args.embeds, dev, torch.bfloat16)
    env = W.env_stamp()

    t0 = time.time()
    for i, clip in enumerate(clips):
        dst = outdir / (clip["clip_id"].replace("/", "__") + ".pt")
        if dst.exists():
            continue
        try:
            frames = read_spanning(clip["path"], N_FRAMES)
        except Exception as exc:
            print(f"SKIP {clip['clip_id']}: {exc}", flush=True)
            continue

        px = W.preprocess_frames(frames, SIZE, SIZE, dev)
        z0 = W.encode_latents(vae, px, normalize=True)
        grid = W.latent_grid(z0)

        rec = {
            "clip": clip, "env": env, "preflight": pre,
            "backbone": args.backbone, "dataset": "physics_iq",
            "grid": list(grid),
            "config": {"height": SIZE, "width": SIZE, "frames": N_FRAMES,
                       "sampling": "uniform over whole clip",
                       "t_grid": [T_FIXED], "feat_seeds": FEAT_SEEDS,
                       "n_layers": len(layers), "tag": "e21_physics_iq"},
            "vae_latent_pooled": z0[0].mean(dim=(2, 3)).transpose(0, 1).half().cpu(),
            "raw_pixel_32": pooled_raw_pixel(px),
            "feats": {str(T_FIXED): {}},
        }

        for seed in FEAT_SEEDS:
            z_t, _, _ = W.flow_noise(z0, T_FIXED, seed=seed)
            ts = W.per_token_timestep(z_t, T_FIXED)
            with W.capture_blocks(tf, layers, grid) as hs:
                W.forward_v(tf, z_t, ts, pe)
            rec["feats"][str(T_FIXED)][str(seed)] = torch.stack(
                [hs.pooled[k] for k in layers], dim=0).half()

        torch.save(rec, dst)
        if (i + 1) % 25 == 0 or i == 0:
            el = time.time() - t0
            print(f"[{i+1}/{len(clips)}] {el:.0f}s ({el/(i+1):.2f}s/clip) "
                  f"peak={torch.cuda.max_memory_allocated()/2**30:.1f}GB", flush=True)

    print(f"shard {args.shard} done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
