"""E12 extraction: the same backbone at different input geometry, i.e. different token budgets.

Why a separate script: E8 held the geometry fixed at 512x512x49 across every backbone, and
`e8_extract*.py` hardcode SIZE/N_FRAMES (e8_extract.py asserts the latent grid is exactly
13x32x32).  E12's whole point is to move that geometry *within one model*, so the constants
become arguments.  Everything else — the record layout, the probe, the lower bounds — is
unchanged, so `e8_analyse.py` machinery still reads these records.

Two axes, and they are NOT interchangeable (see E12_PREREG.md §4):

  frames   LTX only.  Changes F_lat, therefore changes BOTH the token count and the feature
           dimension F_lat*D.  Reported under two calibers: flattened (as E8) and
           time-averaged (dimension held constant).  Only the latter isolates the budget.
  size     CogVideoX only.  F_lat stays 13, so the feature dimension is constant and the
           token count is the only thing that moves.  But it also changes spatial detail,
           which is why it cannot carry the Q1 verdict alone.

Records go to their own cache dir per geometry, so no existing E8 cache is touched:

  cache/<prefix>_feats_<weights>_<dataset>[_cj]_<tag>     tag defaults to S<size>F<frames>

Run (one process per GPU):
  HF_HOME=... CUDA_VISIBLE_DEVICES=i python scripts/e12_extract.py \
      --model ltx --size 512 --frames 193 --dataset intphys2 --weights real \
      --variant plain --shard i --num-shards 4
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
OUT = ROOT / "results" / "E12_token_budget"
FEAT_SEEDS = [0, 1, 2]

MANIFESTS = {
    "intphys2": ROOT / "results" / "E7_intphys2" / "manifest.json",
    "likephys": ROOT / "results" / "T3_null_floor" / "manifest.json",
}

MODELS = {
    # prefix must match e8_analyse.BACKBONES so the record is readable there
    "ltx": {
        "prefix": "ltx",
        "repo": "Lightricks/LTX-Video",
        "revision": "8984fa25007f376c1a299016d0957a37a2f797bb",
        "family": "ltx",
        "n_layers": 28,
        "prompt_embeds": "prompt_embeds_empty_cogvideox.pt",
    },
    "cogvideox2b": {
        "prefix": "cogvideox",
        "repo": "zai-org/CogVideoX-2b",
        "revision": "1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01",
        "family": "cogvideox",
        "n_layers": 30,
        "prompt_embeds": "prompt_embeds_empty_cogvideox.pt",
    },
}


def snapshot(m: dict) -> Path:
    import os

    d = (Path(os.environ["HF_HOME"]) / "hub"
         / ("models--" + m["repo"].replace("/", "--")) / "snapshots" / m["revision"])
    if not d.exists():
        raise FileNotFoundError(f"{d} missing — download {m['repo']} first")
    return d


def out_dir(m: dict, weights: str, dataset: str, variant: str, tag: str) -> Path:
    suffix = "" if variant == "plain" else f"_{variant}"
    return ROOT / "cache" / f"{m['prefix']}_feats_{weights}_{dataset}{suffix}_{tag}"


def check_geometry(family: str, size: int, frames: int) -> None:
    """Fail before loading a 20 GB model, not after."""
    if family == "cogvideox":
        assert size % 8 == 0, "CogVideoX VAE downsamples 8x spatially"
        assert (frames - 1) % 4 == 0, "CogVideoX VAE downsamples 4x temporally"
    elif family == "ltx":
        assert size % 32 == 0, "LTX VAE downsamples 32x spatially"
        assert (frames - 1) % 8 == 0, "LTX VAE downsamples 8x temporally"


# ---------------------------------------------------------------------------------------
# per-family encode / forward, mirroring e8_extract*.py but with geometry as arguments
# ---------------------------------------------------------------------------------------
def build(family: str, snap: Path, dev: str, random_init: bool):
    if family == "ltx":
        import e8_extract_ltx as L
        from diffusers import AutoencoderKLLTXVideo, LTXVideoTransformer3DModel

        vae = AutoencoderKLLTXVideo.from_pretrained(
            str(snap / "vae"), torch_dtype=torch.float32).to(dev).eval()
        src = str(snap / "transformer")
        if random_init:
            cfg = LTXVideoTransformer3DModel.load_config(src)
            with torch.device("meta"):
                tf = LTXVideoTransformer3DModel.from_config(cfg)
            tf = tf.to_empty(device=dev)
            L._init_norm_aware(tf)
            tf = tf.to(torch.bfloat16).eval()
        else:
            tf = LTXVideoTransformer3DModel.from_pretrained(
                src, torch_dtype=torch.bfloat16).to(dev).eval()
        return vae, tf, L.encode, L.latent_grid, L.forward_v, L.capture, L.pooled_raw_pixel

    from diffusers import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel

    vae = AutoencoderKLCogVideoX.from_pretrained(
        str(snap / "vae"), torch_dtype=torch.float32).to(dev).eval()
    src = str(snap / "transformer")
    if random_init:
        cfg = CogVideoXTransformer3DModel.load_config(src)
        with torch.device("meta"):
            tf = CogVideoXTransformer3DModel.from_config(cfg)
        tf = tf.to_empty(device=dev)
        C._init_weights_norm_aware(tf)
        tf = tf.to(torch.bfloat16).eval()
    else:
        tf = CogVideoXTransformer3DModel.from_pretrained(
            src, torch_dtype=torch.bfloat16).to(dev).eval()

    def _pool_raw(px, f_lat):
        # CogVideoX groups by 4 latent frames (its temporal compression), like E7/E8
        return C.pooled_raw_pixel(px)

    return vae, tf, C.encode_latents, C.latent_grid, None, C.capture_blocks, _pool_raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--frames", type=int, required=True)
    ap.add_argument("--dataset", choices=list(MANIFESTS), default="intphys2")
    ap.add_argument("--weights", choices=["real", "random"], default="real")
    ap.add_argument("--variant", choices=["plain", "cj"], default="plain")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--t-grid", type=int, nargs="+", default=None)
    ap.add_argument("--tag", default="", help="cache tag; defaults to S<size>F<frames>")
    args = ap.parse_args()

    m = MODELS[args.model]
    check_geometry(m["family"], args.size, args.frames)
    if args.dataset == "likephys" and args.variant == "cj":
        raise SystemExit("LikePhys carries its own color_change clips; no jitter variant needed")

    tag = args.tag or f"S{args.size}F{args.frames}"
    t_grid = args.t_grid or [600]

    dev = "cuda:0"
    pre = W.preflight(dev)

    man = json.loads(MANIFESTS[args.dataset].read_text())
    clips = man if args.variant == "plain" else [c for c in man if c["label"] == 0]
    clips = [c for i, c in enumerate(clips) if i % args.num_shards == args.shard]
    if args.limit:
        clips = clips[: args.limit]

    od = out_dir(m, args.weights, args.dataset, args.variant, tag)
    od.mkdir(parents=True, exist_ok=True)
    print(f"e12/{args.model}/{args.weights}/{args.dataset}/{args.variant}/{tag} "
          f"shard {args.shard}/{args.num_shards}: {len(clips)} clips -> {od}", flush=True)
    print("preflight:", json.dumps(pre), flush=True)

    vae, tf, encode, latent_grid, forward_v, capture, pool_raw = build(
        m["family"], snapshot(m), dev, random_init=(args.weights == "random"))
    n_layers = tf.config.num_layers
    if m["family"] == "ltx":
        # LTX's forward needs the (F,H,W) triple for its internal RoPE; bind it here so the
        # call site below is identical for both families.
        def _forward(g):
            import e8_extract_ltx as L

            def f(z_t, t, pe):
                return L.forward_v(tf, z_t, t, pe, g)
            return f
    else:
        def _forward(g):
            def f(z_t, t, pe):
                return C.forward_v(tf, z_t, torch.tensor([t], device=z_t.device), pe, grid=g)
            return f

    layers = list(range(n_layers))
    # The two capture helpers return different containers: LTX yields a plain dict
    # {layer: [F_lat, D]}, CogVideoX/Wan yield a HiddenStates object whose `pooled` holds the
    # same thing. Normalised here so the per-clip loop has one accessor.
    def take(hs, k):
        return hs.pooled[k] if hasattr(hs, "pooled") else hs[k]

    pe, _ = W.load_prompt_embeds(OUT.parent / "E8_second_backbone" / m["prompt_embeds"],
                                 dev, torch.bfloat16)
    env = W.env_stamp()

    t0 = time.time()
    grid_seen = None
    for i, clip in enumerate(clips):
        cid = clip.get("clip_id", "")
        dst = od / (cid.replace("/", "__") + ".pt")
        if dst.exists():
            continue
        try:
            frames = C.read_spanning(clip["path"], args.frames, args.size)
        except Exception as exc:
            print(f"SKIP {cid}: {exc}", flush=True)
            continue

        px = W.preprocess_frames(frames, args.size, args.size, dev)
        if args.variant == "cj":
            px = colour_jitter(px)
        z0 = encode(vae, px)
        grid = latent_grid(z0)
        grid_seen = grid
        f_lat, gh, gw = grid
        tokens = f_lat * gh * gw

        raw = pool_raw(px, f_lat)
        vae_pool = z0[0].mean(dim=(2, 3))          # [F_lat, C] for both families
        rec = {
            "clip": clip, "env": env, "preflight": pre,
            "backbone": args.weights, "variant": args.variant, "dataset": args.dataset,
            "grid": list(grid),
            "config": {
                "height": args.size, "width": args.size, "frames": args.frames,
                "tokens": tokens, "sampling": "uniform over whole clip",
                "t_grid": t_grid, "feat_seeds": FEAT_SEEDS,
                "model": m["repo"], "revision": m["revision"], "n_layers": n_layers,
                "family": m["family"], "latent_channels": int(z0.shape[2]),
                "colour_jitter": ({"hue": 0.35, "sat": 1.4}
                                 if args.variant == "cj" else None),
                "e12_tag": tag,
                "tag": f"e12_{args.model}_{args.dataset}_{args.variant}_{tag}",
            },
            "vae_latent_pooled": vae_pool.half().cpu(),
            "raw_pixel_32": raw,
            # Dimension-controlled twins (E12_PREREG.md §4.4). Same [n_layers, D] layout as
            # `feats` but with the temporal axis averaged out, so every rung of a ladder has
            # the same dimensionality and the probe cannot win by having more inputs.
            "vae_latent_timeavg": vae_pool.mean(dim=0).half().cpu(),
            "raw_pixel_timeavg": raw.float().mean(dim=0).half().cpu(),
            "feats": {},
            "feats_timeavg": {},
        }

        fwd = _forward(grid)
        for t in t_grid:
            rec["feats"][str(t)] = {}
            rec["feats_timeavg"][str(t)] = {}
            for seed in FEAT_SEEDS:
                if m["family"] == "ltx":
                    g_ = torch.Generator(device=z0.device).manual_seed(int(seed))
                    eps = torch.randn(z0.shape, generator=g_, device=z0.device,
                                      dtype=torch.float32)
                    sigma = float(t) / 1000.0
                    z_t = ((1.0 - sigma) * z0 + sigma * eps).to(z0.dtype)
                else:
                    z_t, _eps, _v = C.flow_noise(z0, t, seed=seed)
                with capture(tf, layers, grid) as hs:
                    fwd(z_t, t, pe)
                stacked = torch.stack([take(hs, k) for k in layers], dim=0).half()  # [L, F_lat, D]
                rec["feats"][str(t)][str(seed)] = stacked
                rec["feats_timeavg"][str(t)][str(seed)] = stacked.float().mean(dim=1).half()

        torch.save(rec, dst)
        if (i + 1) % 25 == 0 or i == 0:
            el = time.time() - t0
            print(f"[{i+1}/{len(clips)}] {el:.0f}s ({el/(i+1):.2f}s/clip) grid={grid} "
                  f"tok={tokens} peak={torch.cuda.max_memory_allocated()/2**30:.1f}GB",
                  flush=True)

    print(f"shard {args.shard} done in {time.time()-t0:.0f}s grid={grid_seen}", flush=True)


if __name__ == "__main__":
    main()
