"""E13 extraction: a NON-generative video encoder (V-JEPA 2) under the same probe.

Why a separate script: every other backbone in this project is a denoising diffusion
transformer.  V-JEPA 2 has no VAE, no noising schedule and no text conditioning, so three
things E8/E12 took for granted simply do not exist:

  no timestep   D1's contrast is real-vs-random-init on CLEAN video.  That is NOT the same
                quantity as E8's D1 (which reads a noised latent), so the two are never
                placed in the same numeric table — only the verdict directions are compared
                (E13_PREREG.md §3.1).
  no VAE        The "VAE latent" lower bound cannot exist.  Reported as explicitly missing
                rather than substituted with something else (§3.3).
  fixed geometry The position encoding is built for 64 frames @ 256x256.  Feeding anything
                else either errors or silently interpolates — and a silently wrong
                positional encoding is exactly the CogVideoX-5b RoPE failure in E8 §12.
                Geometry is therefore asserted, not parameterised.

Per-frame pooling matches the project's existing record layout: V-JEPA 2's tubelet is 2, so
64 frames -> 32 temporal positions; hidden states are [1, 8192, 1024] = 32 x 256 x 1024, and
mean-pooling the 256 spatial tokens gives [32, 1024] — the same "temporal position x channels"
shape E8 stores for its latent frames, so `probe` reads these records unchanged.

Run (one process per GPU):
  HF_HOME=... CUDA_VISIBLE_DEVICES=i python scripts/e13_extract.py \
      --dataset intphys2 --weights real --variant plain --shard i --num-shards 4
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wan_probe_lib as W
from e8_extract import colour_jitter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E13_non_generative"
FEAT_SEEDS = [0, 1, 2]

# Geometry is the model's own; see the docstring. Do not parameterise these.
# Two sizes exist for the E13 scale axis; both are 64 frames @ 256 so the geometry block below
# applies to both and the scale comparison is clean (same tokens, same preprocessing).
MODELS = {
    "vitl": {"repo": "facebook/vjepa2-vitl-fpc64-256",
             "revision": "b3c1679b7c34d3255ef3547f27c7b226aefab26f",
             "prefix": "vjepa", "layers": 24},
    "vith": {"repo": "facebook/vjepa2-vith-fpc64-256",
             "revision": "b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5",
             "prefix": "vjepa_h", "layers": 32},
}
REPO = MODELS["vitl"]["repo"]
REVISION = MODELS["vitl"]["revision"]
SIZE = 256
FRAMES = 64
RESIZE_TO = 292          # video_preprocessor_config.json: size.shortest_edge
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

MANIFESTS = {
    "intphys2": ROOT / "results" / "E7_intphys2" / "manifest.json",
    "likephys": ROOT / "results" / "T3_null_floor" / "manifest.json",
}


def snapshot(m: dict) -> Path:
    import os

    d = (Path(os.environ["HF_HOME"]) / "hub"
         / ("models--" + m["repo"].replace("/", "--")) / "snapshots" / m["revision"])
    if not d.exists():
        raise FileNotFoundError(f"{d} missing — download {m['repo']} first")
    return d


def read_clip(path: str) -> np.ndarray:
    """Decode FRAMES spanning the whole clip at RESIZE_TO, then centre-crop to SIZE.

    Spanning rather than first-N: the violation is not annotated in IntPhys 2, so the first
    64 native frames would cover 10% of the clip and could miss it entirely (same reason as
    every other extractor here).
    """
    import decord

    vr = decord.VideoReader(str(path), width=RESIZE_TO, height=RESIZE_TO)
    idx = np.linspace(0, len(vr) - 1, FRAMES).round().astype(int).tolist()
    frames = vr.get_batch(idx).asnumpy()                      # [F, R, R, 3] uint8
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    x = torch.nn.functional.interpolate(x, size=(SIZE, SIZE), mode="bilinear",
                                        align_corners=False)
    return x


def normalise(x: torch.Tensor, dev: str) -> torch.Tensor:
    """[F,3,H,W] in [0,1] -> [1,F,3,H,W] ImageNet-normalised (model's own preprocessing)."""
    m = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return ((x - m) / s).unsqueeze(0).to(dev)


def denormalise(x: torch.Tensor) -> torch.Tensor:
    """Back to [1,3,F,H,W] in [-1,1] so `colour_jitter` applies to the same pixels E8 jittered."""
    m = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    px = (x * s + m).clamp(0, 1)
    return (px * 2 - 1).permute(0, 2, 1, 3, 4).contiguous()


def pool_temporal(hidden: torch.Tensor, tubelet: int) -> torch.Tensor:
    """[1, T*S, D] -> [T, D] by mean over the S spatial tokens of each temporal position.

    The spatial token count is derived, not hardcoded, so a different patch size fails loudly
    instead of scrambling the temporal axis (the failure mode that would silently destroy
    the physics signal while leaving the appearance signal intact).
    """
    _, seq, d = hidden.shape
    grid = SIZE // 16
    spatial = grid * grid
    t = seq // spatial
    assert t * spatial == seq, f"seq {seq} not divisible by {spatial} spatial tokens"
    return hidden.float().view(t, spatial, d).mean(dim=1)[0] if False else \
        hidden.float()[0].view(t, spatial, d).mean(dim=1)


def _init_norm_aware(model, seed: int = 0) -> None:
    """Same random-control rule as CogVideoX/LTX/Mochi: He for ndim>=2, norm weights 1, rest 0.

    V-JEPA 2 is a ViT with LayerNorm; zeroing `weight` would make every block output 0 and
    the between-clip std exactly 0 (the CogVideoX failure, E8_PREREG.md §3).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    for name, p in model.named_parameters():
        # to_empty() leaves params as leaves that require grad; in-place copy_ on those
        # raises. Nothing here is training, so grad is simply off.
        p.requires_grad_(False)
        if p.ndim >= 2:
            fan_in = int(np.prod(p.shape[1:]))
            std = (2.0 / max(fan_in, 1)) ** 0.5
            p.copy_(torch.empty(p.shape, dtype=torch.float32).normal_(0.0, std, generator=g)
                    .to(p.dtype))
        elif "norm" in name and name.endswith("weight"):
            p.fill_(1.0)
        else:
            p.zero_()
    for _n, b in model.named_buffers():
        if b.is_floating_point():
            b.zero_()


def out_dir(m: dict, weights: str, dataset: str, variant: str, tag: str = "") -> Path:
    suffix = "" if variant == "plain" else f"_{variant}"
    t = f"_{tag}" if tag else ""
    return ROOT / "cache" / f"{m['prefix']}_feats_{weights}_{dataset}{suffix}{t}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(MANIFESTS), default="intphys2")
    ap.add_argument("--weights", choices=["real", "random"], default="real")
    ap.add_argument("--variant", choices=["plain", "cj"], default="plain")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="", help="extra cache-dir suffix (used for verification runs)")
    ap.add_argument("--model", default="vitl", choices=list(MODELS))
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

    m = MODELS[args.model]
    od = out_dir(m, args.weights, args.dataset, args.variant, args.tag)
    od.mkdir(parents=True, exist_ok=True)
    print(f"vjepa({args.model})/{args.weights}/{args.dataset}/{args.variant} shard {args.shard}/"
          f"{args.num_shards}: {len(clips)} clips -> {od}", flush=True)
    print("preflight:", json.dumps(pre), flush=True)

    from transformers import VJEPA2Model

    src = str(snapshot(m))
    if args.weights == "random":
        cfg = VJEPA2Model.config_class.from_pretrained(src)
        with torch.device("meta"):
            tf = VJEPA2Model(cfg)
        tf = tf.to_empty(device=dev)
        _init_norm_aware(tf)
        tf = tf.to(torch.bfloat16).eval()
    else:
        tf = VJEPA2Model.from_pretrained(src, dtype=torch.bfloat16).to(dev).eval()

    n_layers = tf.config.num_hidden_layers
    env = W.env_stamp()

    t0 = time.time()
    for i, clip in enumerate(clips):
        cid = clip.get("clip_id", "")
        dst = od / (cid.replace("/", "__") + ".pt")
        if dst.exists():
            continue
        try:
            x = read_clip(clip["path"])
        except Exception as exc:
            print(f"SKIP {cid}: {exc}", flush=True)
            continue

        if args.variant == "cj":
            # jitter in [-1,1] space on the same pixel grid E8 used, then back to the
            # model's normalisation — so the perturbation is identical across backbones.
            px = denormalise(normalise(x, dev))
            jit = colour_jitter(px)[0]                       # [3, F, H, W] in [-1,1]
            x = ((jit + 1) / 2
                 - torch.tensor(MEAN, device=dev).view(3, 1, 1, 1)) / \
                torch.tensor(STD, device=dev).view(3, 1, 1, 1)
            x = x.permute(1, 0, 2, 3).unsqueeze(0)           # [1, F, 3, H, W]
        else:
            x = normalise(x, dev)

        with torch.no_grad():
            o = tf(pixel_values_videos=x.to(torch.bfloat16), output_hidden_states=True)
        hs = o.hidden_states                          # tuple of n_layers+1 tensors
        pooled = torch.stack([pool_temporal(h, 2) for h in hs[1:]], dim=0).half()  # [L, 32, D]

        rec = {
            "clip": clip, "env": env, "preflight": pre,
            "backbone": args.weights, "variant": args.variant, "dataset": args.dataset,
            "grid": [FRAMES // 2, SIZE // 16, SIZE // 16],
            "config": {"height": SIZE, "width": SIZE, "frames": FRAMES,
                       "resize_to": RESIZE_TO, "tokens": (FRAMES // 2) * (SIZE // 16) ** 2,
                       "sampling": "uniform over whole clip", "t_grid": [],
                       "feature_seeds": [0],        # deterministic: no noising to re-draw
                       "model": m["repo"], "revision": m["revision"], "n_layers": n_layers,
                       "family": "vjepa2", "latent_channels": None,
                       "colour_jitter": ({"hue": 0.35, "sat": 1.4}
                                         if args.variant == "cj" else None),
                       "tag": f"e13_vjepa_{args.dataset}_{args.variant}"},
            # No VAE: the VAE-latent lower bound is absent by construction, recorded as None
            # rather than filled with a substitute (E13_PREREG.md §3.3).
            "vae_latent_pooled": None,
            "raw_pixel_32": raw_pixel_32(denormalise(x)),
            "feats": {"clean": {"0": pooled}},
            "feats_timeavg": {"clean": {"0": pooled.float().mean(dim=1).half()}},
        }
        torch.save(rec, dst)
        if (i + 1) % 25 == 0 or i == 0:
            el = time.time() - t0
            print(f"[{i+1}/{len(clips)}] {el:.0f}s ({el/(i+1):.2f}s/clip) "
                  f"feats={tuple(pooled.shape)} peak={torch.cuda.max_memory_allocated()/2**30:.1f}GB",
                  flush=True)

    print(f"shard {args.shard} done in {time.time()-t0:.0f}s", flush=True)


def raw_pixel_32(px: torch.Tensor, size: int = 32) -> torch.Tensor:
    """[1,3,F,H,W] in [-1,1] -> per-temporal-position 32x32 RGB, grouped like the features.

    Grouping matches V-JEPA's tubelet-2 stride: FRAMES/2 positions, each covering 2 frames.
    """
    x = (px[0].permute(1, 0, 2, 3) + 1.0) * 0.5                       # [F,3,H,W]
    x = torch.nn.functional.interpolate(x, size=(size, size), mode="bilinear",
                                        align_corners=False)
    n = x.shape[0]
    per = n // (FRAMES // 2)
    groups = [x[i * per:(i + 1) * per].mean(dim=0) for i in range(FRAMES // 2)]
    return torch.stack(groups, dim=0).flatten(1).half().cpu()


if __name__ == "__main__":
    main()
