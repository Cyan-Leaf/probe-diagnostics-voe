"""E17 extraction: Wan2.2-T2V-A14B, whose weight package holds TWO independent experts.

What makes this model different from every other backbone here, and therefore what this
script has to get right:

  * **Two 40-layer transformers in one repo.** `transformer` handles high noise (before the
    boundary), `transformer_2` handles low noise (after it). They share architecture, data and
    training recipe, so any difference between them is attributable to the noise range they
    were trained on — the one controlled contrast this project cannot get from two different
    model families.
  * **t=600 sits after the boundary.** `model_index.json` has `boundary_ratio: 0.875`, and at
    t=600 sigma = 0.600 < 0.875, so the *low-noise* expert is the in-range one. Running the
    high-noise expert at t=600 is out-of-range and is declared in E17_PREREG.md §4.1: its
    numbers answer "does a 14B denoiser trained on another noise band carry physics", not
    "how good is this expert in its own regime".
  * **Same VAE as Wan2.1** (16-channel `AutoencoderKLWan`), so the latent layout and the
    pixel bound are identical to the Wan2.1 runs and need no special casing.

One expert is loaded at a time. Both would fit in 183 GB, but loading both would change the
memory conditions relative to every other backbone in the study for no benefit.

Run (one process per GPU):
  HF_HOME=... CUDA_VISIBLE_DEVICES=i python scripts/e17_extract.py \
      --expert low --dataset intphys2 --weights real --variant plain
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
OUT = ROOT / "results" / "E17_moe_experts"
FEAT_SEEDS = [0, 1, 2]

REPO = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
REVISION = "5be7df9619b54f4e2667b2755bc6a756675b5cd7"
N_FRAMES = 49
SIZE = 512
T_GRID = [600]          # default; --t overrides (E18 uses 300 / 900 for the own-band cells)
BOUNDARY_RATIO = 0.875          # from model_index.json, verified on disk
EXPERTS = {"high": "transformer", "low": "transformer_2"}

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


def out_dir(expert: str, weights: str, dataset: str, variant: str, t: int = 600) -> Path:
    """Cache dir carries the timestep when it is not the E17 default 600.

    E18 needs t=300 and t=900 on the SAME expert, so the timestep has to be part of the path;
    otherwise the new cells would silently overwrite E17's delivered t=600 features.
    """
    suffix = "" if variant == "plain" else f"_{variant}"
    tpart = "" if t == 600 else f"_t{t}"
    return ROOT / "cache" / f"wan22a14b_{expert}_feats_{weights}_{dataset}{suffix}{tpart}"


def capture(tf, layers: list[int], grid: tuple[int, int, int]):
    """Unknown-typed Wan block hook; same pooling contract as e8_backbones.capture."""
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
                handles.append(tf.blocks[i].register_forward_hook(make_hook(i)))
            yield store
        finally:
            for h in handles:
                h.remove()

    return ctx()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert", required=True, choices=list(EXPERTS))
    ap.add_argument("--dataset", choices=list(MANIFESTS), default="intphys2")
    ap.add_argument("--weights", choices=["real", "random"], default="real")
    ap.add_argument("--variant", choices=["plain", "cj"], default="plain")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--t", type=int, default=600,
                    help="noising timestep; non-600 writes to a t-suffixed cache dir (E18)")
    args = ap.parse_args()

    if args.dataset == "likephys" and args.variant == "cj":
        raise SystemExit("LikePhys carries its own color_change clips; no jitter variant needed")

    dev = "cuda:0"
    pre = W.preflight(dev)
    t_grid = [args.t]
    sigma = t_grid[0] / 1000.0
    in_range = (sigma >= BOUNDARY_RATIO) if args.expert == "high" else (sigma < BOUNDARY_RATIO)
    if not in_range:
        print(f"NOTE: expert={args.expert} at t={t_grid[0]} (sigma {sigma}) is OUT OF RANGE "
              f"for boundary_ratio={BOUNDARY_RATIO}. Declared in E17_PREREG.md §4.1.",
              flush=True)

    man = json.loads(MANIFESTS[args.dataset].read_text())
    clips = man if args.variant == "plain" else [c for c in man if c["label"] == 0]
    clips = [c for i, c in enumerate(clips) if i % args.num_shards == args.shard]
    if args.limit:
        clips = clips[: args.limit]

    od = out_dir(args.expert, args.weights, args.dataset, args.variant, args.t)
    od.mkdir(parents=True, exist_ok=True)
    print(f"a14b/{args.expert}/{args.weights}/{args.dataset}/{args.variant} shard "
          f"{args.shard}/{args.num_shards}: {len(clips)} clips -> {od}", flush=True)
    print("preflight:", json.dumps(pre), flush=True)

    from diffusers import AutoencoderKLWan, WanTransformer3DModel

    snap = snapshot()
    vae = AutoencoderKLWan.from_pretrained(str(snap / "vae"),
                                           torch_dtype=torch.float32).to(dev).eval()
    src = str(snap / EXPERTS[args.expert])
    if args.weights == "random":
        cfg = WanTransformer3DModel.load_config(src)
        with torch.device("meta"):
            tf = WanTransformer3DModel.from_config(cfg)
        tf = tf.to_empty(device=dev)
        W._init_weights_(tf)
        tf = tf.to(torch.bfloat16).eval()
    else:
        tf = WanTransformer3DModel.from_pretrained(
            src, torch_dtype=torch.bfloat16).to(dev).eval()

    layers = list(range(tf.config.num_layers))
    pe, _ = W.load_prompt_embeds(OUT.parent / "E8_second_backbone" / "prompt_embeds_empty_wan.pt",
                                 dev, torch.bfloat16)
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
        z0 = W.encode_latents(vae, px, normalize=True)
        grid = W.latent_grid(z0)

        rec = {
            "clip": clip, "env": env, "preflight": pre,
            "backbone": args.weights, "variant": args.variant, "dataset": args.dataset,
            "grid": list(grid),
            "config": {"height": SIZE, "width": SIZE, "frames": N_FRAMES,
                       "sampling": "uniform over whole clip", "t_grid": t_grid,
                       "feat_seeds": FEAT_SEEDS, "model": REPO, "revision": REVISION,
                       "n_layers": len(layers), "family": "wan",
                       "expert": args.expert, "expert_subfolder": EXPERTS[args.expert],
                       "boundary_ratio": BOUNDARY_RATIO,
                       "in_range_for_t": bool(in_range),
                       "latent_channels": int(z0.shape[1]),
                       "colour_jitter": ({"hue": 0.35, "sat": 1.4}
                                         if args.variant == "cj" else None),
                       "tag": f"e17_a14b_{args.expert}_{args.dataset}_{args.variant}"},
            "vae_latent_pooled": z0[0].mean(dim=(2, 3)).transpose(0, 1).half().cpu(),
            "raw_pixel_32": C.pooled_raw_pixel(px),
            "feats": {},
        }

        for t in t_grid:
            rec["feats"][str(t)] = {}
            for seed in FEAT_SEEDS:
                z_t, _eps, _v = W.flow_noise(z0, t, seed=seed)
                ts = W.per_token_timestep(z_t, t)
                with capture(tf, layers, grid) as hs:
                    W.forward_v(tf, z_t, ts, pe)
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
