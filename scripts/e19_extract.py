"""E19 extraction: MiniMax-H3 (33B) under the E8 protocol -- the size axis.

Why a new script rather than a flag on an existing one
-----------------------------------------------------
Every backbone so far fed the transformer one contiguous video latent. H3 does not work that
way, and getting it wrong fails *silently* (the probe would still return a number):

  * **One packed sequence, three modalities.** Rows are `[text | keyframe conditions | target
    audio | target video]`, and `forward` addresses them through `video_indices` /
    `audio_indices` / `text_indices` plus a per-row `token_tags` and an fp64 `(t, h, w)` rotary
    grid. The rotary clock starts *after* the text rows, so text length shifts every media
    position. Rebuilding this by hand is how a silent offset enters, so `build_packed_sequence`
    and `patchify_video_latents` are imported from diffusers' own H3 layout step -- the same
    rule E11 established for the probe ("never reimplement; import").

  * **The VAE is remote code, not `AutoencoderKLWan`.** It is `MiniMaxH3VideoVAE` (f16t4d24,
    24 latent channels) wrapping an `AutoencoderKLLegacy`, configured through `auto_map`, and
    its `encode` returns raw moments that must be normalised with the checkpoint's own
    `latents_mean` / `latents_std`. Pixel-space normalisation is also model-specific
    (`pixel_mean` / `pixel_std` on the pipeline components) rather than Wan's simple [-1,1].

  * **It predicts video *and* audio latents.** Audio latents are noise the probe has no
    interest in, but they occupy rows in the packed sequence and carry the `audio_tag`, so they
    must be supplied at the right length even for a video-only measurement.

Scope: this script measures the **video** rows only. Audio rows are passed, correctly shaped
and tagged, because `forward` requires them; nothing is read back from them.

Run (one process per GPU):
  .venv_h3/bin/python scripts/e19_extract.py --weights real --variant plain --limit 4
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
OUT = ROOT / "results" / "E19_h3_scale"
FEAT_SEEDS = [0, 1, 2]

REPO = "MiniMaxAI/MiniMax-H3"
# 39 pixel frames, not 49. H3's VAE only accepts `17*n + 5` frames (clip_length 17, token_drop
# 3): 39 = 17*2 + 5 -> 12 latent frames, which is the closest match to the 13 latent frames every
# other backbone produces from 49 frames. 49 itself is NOT encodable (it snaps to 56 -> 17
# latent frames), and silently letting the VAE round it would change the temporal resolution
# relative to E8 for no reason.
N_FRAMES_REQUEST = 49
N_FRAMES = 39
SIZE = 512
T_GRID = [600]          # flow matching, same convention as Wan / LTX / Mochi / Hunyuan
PATCH_SIZE = (1, 2, 2)  # (t, h, w), from transformer/config.json

MANIFESTS = {
    "intphys2": ROOT / "results" / "E7_intphys2" / "manifest.json",
    "likephys": ROOT / "results" / "T3_null_floor" / "manifest.json",
}


def h3_root() -> Path:
    """Where the parallel downloader put FL2VA/. Not the hf-hub snapshot layout."""
    d = Path("$HF_HOME/minimax_h3")
    if not (d / "FL2VA" / "transformer").exists():
        raise FileNotFoundError(f"{d}/FL2VA/transformer missing -- run scripts/piq_par_download.py")
    return d


def out_dir(weights: str, dataset: str, variant: str) -> Path:
    suffix = "" if variant == "plain" else f"_{variant}"
    return ROOT / "cache" / f"minimax_h3_feats_{weights}_{dataset}{suffix}"


def capture(tf, layers: list[int], rows_per_frame: int, num_latent_frames: int):
    """Hook the video rows of the packed sequence and pool them to one vector per latent frame.

    The packed sequence holds text, audio and video rows; only `[video_start:]` is video. Rather
    than trust a hard-coded offset, the hook slices off the *last* `num_latent_frames *
    rows_per_frame` rows, which is where the layout step puts the target video block, and
    asserts that the text+audio prefix is the only thing before it.
    """
    import contextlib

    store = {}
    handles = []
    num_video_rows = num_latent_frames * rows_per_frame

    def make_hook(idx):
        def hook(_m, _i, output):
            h = output[0] if isinstance(output, tuple) else output
            if h.ndim == 2:
                h = h.unsqueeze(0)
            b, seq, d = h.shape
            assert seq >= num_video_rows, f"seq {seq} < video rows {num_video_rows}"
            vid = h[:, seq - num_video_rows:, :]
            store[idx] = vid.float().view(b, num_latent_frames, rows_per_frame, d) \
                            .mean(dim=2)[0].detach().cpu()
        return hook

    @contextlib.contextmanager
    def ctx():
        try:
            for i in layers:
                handles.append(tf.transformer_blocks[i].register_forward_hook(make_hook(i)))
            yield store
        finally:
            for h in handles:
                h.remove()

    return ctx()


@torch.no_grad()
def encode_latents(vae, px_minus1_1: torch.Tensor, lat_mean, lat_std, seed: int = 42):
    """[-1,1] pixels -> normalised latents, following the checkpoint's own recipe.

    Three details differ from every other backbone here, and each is silent if skipped:

      1. **ImageNet normalisation, not [-1,1].** `source/config.json` has
         `pixel_norm_type: "imagenet"`, so the encoder expects `u/255 - mean)/std` with the
         ImageNet constants. Feeding Wan's [-1,1] convention would still produce a latent —
         just not the one this VAE was trained to invert.
      2. **The posterior is sampled, not taken at the mode**, under a generator seeded
         independently of the request (`encode_seed = 42` in the released recipe).
      3. **The sampled latent is rounded to float16 before normalisation.** That is ~11 bits of
         mantissa per latent value; skipping it changes the latent slightly and, more
         importantly, would make our numbers not comparable to the released pipeline.

    `MiniMaxH3VideoVAE.encode` (the FL2VA wrapper) returns raw moments (2C channels: mean then
    logvar), while the top-level diffusers `AutoencoderKLMiniMaxH3.encode` returns an
    `AutoencoderKLOutput` carrying a proper `latent_dist`. Both paths are handled here so the
    two checkpoints -- which are the same VAE saved two ways -- give the same latent.
    """
    x01 = (px_minus1_1 + 1.0) * 0.5                       # [-1,1] -> [0,1]
    mean = torch.tensor([0.485, 0.456, 0.406], device=x01.device).view(1, -1, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x01.device).view(1, -1, 1, 1, 1)
    pixels = (x01.to(torch.float32) - mean) / std

    gen = torch.Generator(device=pixels.device).manual_seed(seed)
    out = vae.encode(pixels)
    if hasattr(out, "latent_dist"):
        z = out.latent_dist.sample(generator=gen)
    else:
        moments = out[0] if isinstance(out, (tuple, list)) else out
        mu, logvar = torch.chunk(moments.float(), 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        noise = torch.randn(mu.shape, generator=gen, device=mu.device, dtype=mu.dtype)
        z = mu + torch.exp(0.5 * logvar) * noise
    z = z.to(torch.float16).float()
    return (z - lat_mean.view(1, -1, 1, 1, 1)) / lat_std.view(1, -1, 1, 1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(MANIFESTS), default="intphys2")
    ap.add_argument("--weights", choices=["real", "random"], default="real")
    ap.add_argument("--variant", choices=["plain", "cj"], default="plain")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--layers", default="", help="comma list; default = reference relative depths")
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
    print(f"h3/{args.weights}/{args.dataset}/{args.variant} shard {args.shard}/{args.num_shards}: "
          f"{len(clips)} clips -> {od}", flush=True)
    print("preflight:", json.dumps(pre), flush=True)

    root = h3_root()
    fl = root / "FL2VA"

    from diffusers import MiniMaxH3Transformer3DModel
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3PrepareLayoutStep, patchify_video_latents)

    # ---- VAE (remote code) -------------------------------------------------------------
    # Two things differ from every other backbone here, and both fail opaquely if guessed:
    #   * `AutoencoderKLMiniMaxH3.from_pretrained` looks for diffusers' default weight filename
    #     and raises, because this component keeps its weights at `source/model.safetensors`
    #     and reads them through its own `from_pretrained`. Loading the class directly is the
    #     supported route, and the local directory has to be importable as a *package* so the
    #     bundle's relative imports resolve -- diffusers' dynamic loader refuses a local path
    #     (`validate_repo_id`), so a namespace package is registered by hand.
    #   * `latents_mean` / `latents_std` live in the *component* config.json, not in the
    #     inner AutoencoderKLLegacy config, so they are read from disk here.
    import importlib
    import types

    # Prefer the top-level diffusers-format `vae/` when it is complete: it is a first-class
    # `AutoencoderKLMiniMaxH3` whose `encode` returns a proper `latent_dist`, and it carries the
    # same ImageNet pixel convention. The FL2VA bundle (raw checkpoint + remote wrapper) stays
    # as a fallback because it was the first thing downloaded.
    top_vae = root / "vae"
    if (top_vae / "config.json").exists() and list(top_vae.glob("*.safetensors")) \
            and (top_vae / "diffusion_pytorch_model.safetensors.index.json").exists():
        from diffusers import AutoencoderKLMiniMaxH3
        vae = AutoencoderKLMiniMaxH3.from_pretrained(str(top_vae),
                                                     torch_dtype=torch.float32).to(dev).eval()
        vae_cfg = json.loads((top_vae / "config.json").read_text())
        vae_kind = "diffusers:AutoencoderKLMiniMaxH3"
    else:
        vae_dir = fl / "video_vae"
        pkg = types.ModuleType("h3vae")
        pkg.__path__ = [str(vae_dir)]
        sys.modules.setdefault("h3vae", pkg)
        vae_mod = importlib.import_module("h3vae.minimax_h3_video_vae")
        vae = vae_mod.MiniMaxH3VideoVAE.from_pretrained(str(vae_dir))
        vae = vae.to(dev).eval() if hasattr(vae, "to") else vae
        vae_cfg = json.loads((vae_dir / "config.json").read_text())
        vae_kind = "fl2va:MiniMaxH3VideoVAE"
    lat_mean = torch.tensor(vae_cfg["latents_mean"], dtype=torch.float32, device=dev)
    lat_std = torch.tensor(vae_cfg["latents_std"], dtype=torch.float32, device=dev)
    print("vae:", vae_kind, flush=True)

    # ---- transformer -------------------------------------------------------------------
    # The top-level `transformer/` is diffusers format (sharded `diffusion_pytorch_model-*`).
    # The FL2VA one is the original checkpoint whose keys are `blocks.*` / `final_layer.*`,
    # which `MiniMaxH3Transformer3DModel.from_pretrained` cannot load at all -- it looks for
    # `diffusion_pytorch_model.safetensors` and would silently fall back to pickle.
    tf_dir = root / "transformer"
    if not (tf_dir / "diffusion_pytorch_model.safetensors.index.json").exists():
        raise SystemExit(
            f"{tf_dir} is missing the diffusers-format index; the FL2VA checkpoint uses the "
            f"original `blocks.*` key layout and is not loadable by diffusers. Download "
            f"`transformer/` from the repo root first.")
    if args.weights == "random":
        # The random-init control must be the SAME architecture with the SAME initialisation
        # rule as every other backbone (Wan's `_init_weights_`, seed 0): Kaiming-uniform std on
        # every matrix, zeros on biases and buffers. Building it on the meta device and filling
        # it with `to_empty` avoids materialising 33B of trained weights we are about to
        # overwrite -- at 62 GB of shards that would cost minutes and 60+ GB of host RAM.
        cfg = MiniMaxH3Transformer3DModel.load_config(str(tf_dir))
        with torch.device("meta"):
            tf = MiniMaxH3Transformer3DModel.from_config(cfg)
        tf = tf.to_empty(device=dev)
        W._init_weights_(tf)
        # H3 pins `proj_in` / `time_embedder` / `rope` etc. to fp32; cast only what is bf16 in
        # the released checkpoint, which is every block weight.
        if hasattr(tf, "_keep_in_fp32_modules"):
            keep = set(tf._keep_in_fp32_modules or [])
            for name, p in tf.named_parameters():
                if not any(k in name for k in keep):
                    p.data = p.data.to(torch.bfloat16)
        tf = tf.eval()
        print("random-init: reinitialised with W._init_weights_(seed=0), zeroed buffers",
              flush=True)
    else:
        tf = MiniMaxH3Transformer3DModel.from_pretrained(
            str(tf_dir), torch_dtype=torch.bfloat16).to(dev).eval()

    n_layers = len(tf.transformer_blocks)
    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    else:
        # Same convention as every other backbone: L16 plus the E8 relative-depth cell.
        layers = sorted({16, int(round(16 / 30.0 * n_layers))})

    # `17*n + 5` pixel frames -> `5*n + 2` latent frames (NOT the /4 the other backbones use;
    # H3's temporal compression is 4x with 3 trailing latents dropped, which does not reduce to
    # a single divisor). 39 = 17*2 + 5 -> 12 latent frames. Taken from diffusers' own function
    # rather than re-derived -- the naive `(N-5)//4+1` gives 9 here, which is wrong.
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import video_latent_num_frames
    latent_frames = video_latent_num_frames(N_FRAMES, 17, 5)
    lh = lw = SIZE // 16

    # Text rows: one tag row. `text_token_tags` only marks modality, so a single text row is
    # enough for the layout; the probe never reads text output.
    text_token_tags = torch.tensor([1], dtype=torch.long)
    num_audio_latents = 8                              # short, video-only measurement
    position_ids, token_tags, video_indices, audio_indices, text_indices, n_cond, _ = \
        MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            text_token_tags=text_token_tags, num_latent_frames=latent_frames,
            latent_height=lh, latent_width=lw, num_audio_latents=num_audio_latents,
            patch_size=PATCH_SIZE, audio_channels=2, audio_tag=2, video_tag=0,
            keyframe_anchors=())

    rows_per_frame = (lh // PATCH_SIZE[1]) * (lw // PATCH_SIZE[2])
    num_video_rows = latent_frames * rows_per_frame
    in_ch = 24 * PATCH_SIZE[0] * PATCH_SIZE[1] * PATCH_SIZE[2]
    text_dim = int(tf.config.text_embed_dim if hasattr(tf.config, "text_embed_dim") else 5120)

    env = W.env_stamp()
    print(f"layout: latent_frames={latent_frames} rows/frame={rows_per_frame} "
          f"video_rows={num_video_rows} seq={len(token_tags)} in_ch={in_ch} layers={layers}",
          flush=True)

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
        z0 = encode_latents(vae, px, lat_mean, lat_std)          # [1,24,12,lh,lw]

        rec = {
            "clip": clip, "env": env, "preflight": pre,
            "backbone": args.weights, "variant": args.variant, "dataset": args.dataset,
            "config": {"height": SIZE, "width": SIZE, "frames": N_FRAMES,
                       "sampling": "uniform over whole clip", "t_grid": T_GRID,
                       "feat_seeds": FEAT_SEEDS, "model": REPO,
                       "n_layers": n_layers, "layers": layers, "family": "minimax_h3",
                       "params_b": 33.1, "latent_channels": 24,
                       "latent_frames": latent_frames, "rows_per_frame": rows_per_frame,
                       "patch_size": list(PATCH_SIZE), "vae": "f16t4d24",
                       "colour_jitter": ({"hue": 0.35, "sat": 1.4}
                                         if args.variant == "cj" else None),
                       "tag": f"e19_h3_{args.dataset}_{args.variant}"},
            "raw_pixel_32": C.pooled_raw_pixel(px),
            # Per-latent-frame spatial mean of the VAE latent. This is the "vae_latent" arm of
            # the four E8 lower bounds; without it the analysis would have to re-encode every
            # clip, doubling the GPU cost of that arm for no information.
            # z0 is [1, 24, latent_frames(12), 32, 32]; after dropping the batch dim the spatial
            # axes are 2 and 3, leaving [24, 12] to transpose into [12, 24]. Verified by encoding
            # one clip: shape (1, 24, 12, 32, 32), not a 4-D latent like Wan's.
            # Field name matches E8's `vae_latent_pooled` so the same reader works on both.
            "vae_latent_pooled": z0[0].mean(dim=(2, 3)).transpose(0, 1).half().cpu(),
            "feats": {},
        }

        # Patchify + pack: video rows come from the latent, audio rows are noise.
        # `patchify_video_latents` drops the batch dimension for B=1 and returns [N, in_ch];
        # `forward` wants [B, N, in_ch], so add it back.
        vid_rows = patchify_video_latents(z0, PATCH_SIZE)
        if vid_rows.ndim == 2:
            vid_rows = vid_rows.unsqueeze(0)
        assert vid_rows.shape[1] == num_video_rows, (vid_rows.shape, num_video_rows)

        for t in T_GRID:
            rec["feats"][str(t)] = {}
            for seed in FEAT_SEEDS:
                g = torch.Generator(device=dev).manual_seed(seed)
                # flow-matching noise on the video latent, same definition as Wan's flow_noise
                eps = torch.randn(z0.shape, generator=g, device=dev, dtype=z0.dtype)
                sigma = t / 1000.0
                z_t = (1.0 - sigma) * z0 + sigma * eps
                z_t_rows = patchify_video_latents(z_t.to(torch.bfloat16), PATCH_SIZE)
                if z_t_rows.ndim == 2:
                    z_t_rows = z_t_rows.unsqueeze(0)

                audio_rows = torch.zeros(1, num_audio_latents * 2, 32,
                                         device=dev, dtype=torch.bfloat16)
                text_rows = torch.zeros(1, 1, text_dim, device=dev, dtype=torch.bfloat16)
                ts = torch.tensor([sigma], device=dev, dtype=torch.bfloat16)
                ts_idx = torch.zeros(len(token_tags), device=dev, dtype=torch.long)

                with capture(tf, layers, rows_per_frame, latent_frames) as hs:
                    tf(hidden_states=z_t_rows, audio_hidden_states=audio_rows,
                       encoder_hidden_states=text_rows, timestep=ts,
                       timestep_indices=ts_idx, token_tags=token_tags.to(dev),
                       position_ids=position_ids.to(dev), video_indices=video_indices.to(dev),
                       audio_indices=audio_indices.to(dev), text_indices=text_indices.to(dev))
                rec["feats"][str(t)][str(seed)] = torch.stack(
                    [hs[k] for k in layers], dim=0).half()

        torch.save(rec, dst)
        if (i + 1) % 5 == 0 or i == 0:
            el = time.time() - t0
            print(f"[{i+1}/{len(clips)}] {el:.0f}s ({el/(i+1):.2f}s/clip) "
                  f"peak={torch.cuda.max_memory_allocated()/2**30:.1f}GB", flush=True)

    print(f"shard {args.shard} done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
