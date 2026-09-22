"""E7: build the IntPhys 2 manifest and extract features (real / random, +/- colour jitter).

Protocol is frozen in results/E7_intphys2/E7_PREREG.md, written before any extraction.
The parts that matter here:

  split ......... Main only (1012 clips, 253 scenes x 4); HeldOut has no public metadata
                  and Debug is a calibration set, not an evaluation set
  labels ........ Impossible -> 1, Possible -> 0  (matches LikePhys, where label 1 = violation)
  grouping ...... SceneIndex, the official pairing key
  frames ........ 49 sampled uniformly across the WHOLE clip (stride = n // 49).
                  IntPhys 2 clips are 636 frames @ 60 fps and the violation timestamp is not
                  annotated, so the first 49 native frames would cover 7.7% of the clip and
                  would likely miss the event entirely.  Punzo et al. (2606.09646, Table 4)
                  likewise use 16 frames per clip.
  colour jitter . D2 control, applied to Possible clips only: one fixed HSV shift
                  (hue +0.35 turn, saturation x1.4) applied identically to every frame, so
                  the physics is untouched and no temporal artefact is introduced

Run:
  python scripts/e7_extract.py --stage manifest
  CUDA_VISIBLE_DEVICES=i python scripts/e7_extract.py --stage extract \
      --backbone real --shard i --num-shards 4
  ... --variant cj      # the colour-jittered control
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import wan_probe_lib as W

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E7_intphys2"
IP_ROOT = Path(os.environ.get(
    "INTPHYS2_ROOT",
    "$HF_HOME/hub/"
    "datasets--facebook--IntPhys2/snapshots/a077a2f94e25889016fc6e5983cf21e2ddc25fb2"))
N_FRAMES = 49
SIZE = 512
T_FIXED = 600
FEAT_SEEDS = [0, 1, 2]

HUE_SHIFT = 0.35
SAT_SCALE = 1.4

# Pre-registered control is "cj".  "cjweak" is an EXPLORATORY sensitivity level added after
# seeing that the registered shift is large (RGB mean abs change 0.30): if a perceptually
# subtle recolour still beats the physics task, D2's failure is not an artefact of picking a
# big shift.  Labelled exploratory per hard rule 4.
JITTER = {
    "cj": {"hue": 0.35, "sat": 1.4},
    "cjweak": {"hue": 0.05, "sat": 1.05},
}


def build_manifest() -> list[dict]:
    src = IP_ROOT / "Main" / "metadata.csv"
    rows = list(csv.DictReader(src.open()))
    out = []
    for r in rows:
        typ = r["type"]
        assert typ in ("1_Possible", "1_Impossible", "2_Possible", "2_Impossible"), typ
        impossible = typ.endswith("Impossible")
        out.append({
            "clip_id": f"intphys2/{r['SceneIndex']}/{typ}/{r['name'][:12]}",
            "path": str(IP_ROOT / "Main" / r["file_name"]),
            "scene_index": r["SceneIndex"],
            "type": typ,
            "label": 1 if impossible else 0,
            "condition": r["condition"],
            "camera": r["Camera"],
            "difficulty": r["Difficulty"],
            "occluder": r["occluder"],
            "game_name": r["game_name"],
            "env": r["env"],
        })
    missing = [c for c in out if not Path(c["path"]).exists()]
    assert not missing, f"{len(missing)} videos missing, e.g. {missing[0]['path']}"
    return out


def read_spanning(path: str, n: int) -> np.ndarray:
    """49 frames spanning the entire clip (see module docstring for why not the first 49)."""
    import decord

    vr = decord.VideoReader(str(path), width=SIZE, height=SIZE)
    total = len(vr)
    idx = np.linspace(0, total - 1, n).round().astype(int).tolist()
    return vr.get_batch(idx).asnumpy()


def colour_jitter(px: torch.Tensor, hue: float = HUE_SHIFT,
                  sat: float = SAT_SCALE) -> torch.Tensor:
    """[1,3,F,H,W] in [-1,1] -> hue/saturation shifted, identical for every frame.

    Physics-preserving by construction: one fixed transform, no temporal component, no
    geometry change.  Semantics match LikePhys `color_change` (recolour, physics legal).
    """
    x = (px[0].permute(1, 0, 2, 3) + 1.0) * 0.5           # [F,3,H,W] in [0,1]
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


def pooled_raw_pixel(px: torch.Tensor, size: int = 32) -> torch.Tensor:
    x = (px[0].permute(1, 0, 2, 3) + 1.0) * 0.5
    x = torch.nn.functional.interpolate(x, size=(size, size), mode="bilinear",
                                        align_corners=False)
    groups = [x[0:1]] + [x[1 + 4 * i: 1 + 4 * (i + 1)] for i in range((x.shape[0] - 1) // 4)]
    return torch.stack([g.mean(dim=0) for g in groups], dim=0).flatten(1).half().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["manifest", "extract"], required=True)
    ap.add_argument("--backbone", choices=["real", "random"], default="real")
    ap.add_argument("--variant", choices=["plain", "cj", "cjweak"], default="plain")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--embeds", default=str(ROOT / "results/T2_env/prompt_embeds_empty.pt"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    mpath = OUT / "manifest.json"

    if args.stage == "manifest":
        man = build_manifest()
        mpath.write_text(json.dumps(man, indent=1, ensure_ascii=False))
        import collections
        print(f"{len(man)} clips, {len({c['scene_index'] for c in man})} scenes")
        for k in ("type", "condition", "camera", "difficulty"):
            print(f"  {k:11s} {dict(collections.Counter(c[k] for c in man))}")
        print(f"  label     {dict(collections.Counter(c['label'] for c in man))}")
        print(f"wrote {mpath}")
        return

    man = json.loads(mpath.read_text())
    clips = man if args.variant == "plain" else [c for c in man if c["label"] == 0]
    clips = [c for i, c in enumerate(clips) if i % args.num_shards == args.shard]
    if args.limit:
        clips = clips[: args.limit]

    dev = "cuda:0"
    pre = W.preflight(dev)
    print("preflight:", json.dumps(pre), flush=True)

    suffix = "" if args.variant == "plain" else f"_{args.variant}"
    outdir = Path(args.outdir or ROOT / "cache" / f"feats_{args.backbone}_intphys2{suffix}")
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"{args.backbone}/{args.variant} shard {args.shard}/{args.num_shards}: "
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
        if args.variant != "plain":
            j = JITTER[args.variant]
            px = colour_jitter(px, hue=j["hue"], sat=j["sat"])
        z0 = W.encode_latents(vae, px, normalize=True)
        grid = W.latent_grid(z0)

        rec = {
            "clip": clip,
            "env": env,
            "preflight": pre,
            "backbone": args.backbone,
            "variant": args.variant,
            "grid": list(grid),
            "config": {"height": SIZE, "width": SIZE, "frames": N_FRAMES,
                       "sampling": "uniform over whole clip",
                       "t_grid": [T_FIXED], "feat_seeds": FEAT_SEEDS,
                       "colour_jitter": (JITTER[args.variant]
                                         if args.variant != "plain" else None),
                       "n_layers": len(layers), "tag": f"e7_intphys2_{args.variant}"},
            "vae_latent_pooled": z0[0].mean(dim=(2, 3)).transpose(0, 1).half().cpu(),
            "raw_pixel_32": pooled_raw_pixel(px),
            "errors": {}, "feats": {},
        }

        rec["feats"][str(T_FIXED)] = {}
        rec["errors"][str(T_FIXED)] = {}
        for seed in FEAT_SEEDS:
            z_t, _, v_tgt = W.flow_noise(z0, T_FIXED, seed=seed)
            ts = W.per_token_timestep(z_t, T_FIXED)
            with W.capture_blocks(tf, layers, grid) as hs:
                v_pred = W.forward_v(tf, z_t, ts, pe)
            rec["feats"][str(T_FIXED)][str(seed)] = torch.stack(
                [hs.pooled[k] for k in layers], dim=0).half()
            e = W.denoising_error(v_pred, v_tgt)
            rec["errors"][str(T_FIXED)][str(seed)] = {
                "mse": e["mse"], "per_frame": [round(x, 6) for x in e["per_frame"]]}

        torch.save(rec, dst)
        if (i + 1) % 25 == 0 or i == 0:
            el = time.time() - t0
            print(f"[{i+1}/{len(clips)}] {el:.0f}s ({el/(i+1):.2f}s/clip) "
                  f"peak={torch.cuda.max_memory_allocated()/2**30:.1f}GB", flush=True)

    print(f"shard {args.shard} done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
