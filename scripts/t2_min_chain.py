"""T2c: minimal chain smoke test — load model -> VAE encode -> single forward -> denoising error.

Also verifies the four MODELS.md traps numerically, because each of them fails *silently*:
  #1 latent normalisation (per-channel mean/std)
  #3 expand_timesteps (per-token vs scalar timestep)
  #5 flow-matching target is eps - x0, not eps
  #8 no text encoder resident (prompt embeds loaded from disk)

Run:
  export HF_HOME=$HF_HOME CUDA_VISIBLE_DEVICES=0
  uv run --no-project python scripts/t2_min_chain.py
"""

import argparse
import glob
import json
import time
from pathlib import Path

import torch

import wan_probe_lib as W

OUT = Path(__file__).resolve().parents[1] / "results" / "T2_env"


def find_likephys_video(rel: str) -> str:
    pat = f"{Path(W.__file__).parent}"  # unused, keep import order tidy
    del pat
    hits = sorted(glob.glob(f"{HUB}/datasets--JianhaoDYDY--LikePhys-Benchmark/snapshots/*/data/{rel}"))
    if not hits:
        raise FileNotFoundError(rel)
    return hits[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="ball_drop_videos/subgroup_000/valid_00.mp4")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--frames", type=int, default=49)  # 4k+1 (risk #10)
    ap.add_argument("--timesteps", type=int, nargs="+", default=[200, 400, 600, 800, 950])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 5, 10, 15, 20, 25, 29])
    ap.add_argument("--embeds", default=str(OUT / "prompt_embeds_empty.pt"))
    ap.add_argument("--out", default=str(OUT / "min_chain.json"))
    args = ap.parse_args()

    dev = "cuda:0"
    rec: dict = {"args": vars(args), "env": W.env_stamp(), "preflight": W.preflight(dev), "steps": []}
    print("preflight:", json.dumps(rec["preflight"]), flush=True)
    torch.cuda.reset_peak_memory_stats()

    # ---- 1. load ---------------------------------------------------------------------
    t0 = time.time()
    vae = W.load_vae(dev, torch.float32)
    rec["steps"].append({"step": "load_vae", "secs": round(time.time() - t0, 1), **W.vram_report("vae")})
    t0 = time.time()
    tf = W.load_transformer(dev, torch.bfloat16)
    rec["steps"].append(
        {"step": "load_transformer_bf16", "secs": round(time.time() - t0, 1), **W.vram_report("vae+dit")}
    )
    n_params = sum(p.numel() for p in tf.parameters())
    rec["model"] = {
        "num_layers": tf.config.num_layers,
        "in_channels": tf.config.in_channels,
        "inner_dim": tf.config.num_attention_heads * tf.config.attention_head_dim,
        "params_B": round(n_params / 1e9, 3),
        "dit_weight_GB_bf16": round(n_params * 2 / 2**30, 2),
        "vae_z_dim": vae.config.z_dim,
        "vae_scale_spatial": vae.config.scale_factor_spatial,
        "vae_scale_temporal": vae.config.scale_factor_temporal,
    }

    pe, pe_blob = W.load_prompt_embeds(args.embeds, dev, torch.bfloat16)
    rec["prompt_embeds"] = {
        "path": args.embeds,
        "shape": list(pe.shape),
        "prompts": pe_blob["prompts"],
        "text_encoder_loaded_now": False,
    }

    # ---- 2. VAE encode ---------------------------------------------------------------
    video = find_likephys_video(args.video)
    frames = W.read_video_frames(video, args.frames)
    px = W.preprocess_frames(frames, args.height, args.width, dev)
    t0 = time.time()
    z0 = W.encode_latents(vae, px, normalize=True)
    z0_raw = W.encode_latents(vae, px, normalize=False)
    enc_secs = round(time.time() - t0, 2)
    grid = W.latent_grid(z0)
    rec["encode"] = {
        "video": video,
        "frames_decoded": list(frames.shape),
        "pixels": list(px.shape),
        "latents": list(z0.shape),
        "token_grid": list(grid),
        "seq_len": grid[0] * grid[1] * grid[2],
        "secs_both_passes": enc_secs,
        "z_normalised": {"mean": z0.mean().item(), "std": z0.std().item(),
                         "min": z0.min().item(), "max": z0.max().item()},
        "z_raw": {"mean": z0_raw.mean().item(), "std": z0_raw.std().item(),
                  "min": z0_raw.min().item(), "max": z0_raw.max().item()},
        **W.vram_report("after_encode"),
    }

    # ---- 3. single forward at each timestep, with hidden-state capture ----------------
    sweep = []
    for t_int in args.timesteps:
        z_t, eps, v_tgt = W.flow_noise(z0, t_int, seed=args.seed)
        ts_tok = W.per_token_timestep(z_t, t_int)
        torch.cuda.synchronize()
        t0 = time.time()
        with W.capture_blocks(tf, args.layers, grid) as hs:
            v_pred = W.forward_v(tf, z_t, ts_tok, pe)
        torch.cuda.synchronize()
        fwd_secs = round(time.time() - t0, 3)

        err = W.denoising_error(v_pred, v_tgt)
        wrong_eps = ((v_pred - eps) ** 2).mean().item()  # risk #5: wrong target
        wrong_x0 = ((v_pred - (-z0)) ** 2).mean().item()  # another wrong target for reference

        # risk #3: scalar timestep instead of per-token
        ts_scalar = torch.tensor([t_int], device=dev, dtype=torch.float32)
        v_pred_scalar = W.forward_v(tf, z_t, ts_scalar, pe)
        scalar_mse = ((v_pred_scalar - v_tgt) ** 2).mean().item()
        rel_diff = ((v_pred_scalar - v_pred).norm() / v_pred.norm()).item()

        sweep.append(
            {
                "t": t_int,
                "sigma": t_int / W.NUM_TRAIN_TIMESTEPS,
                "fwd_secs": fwd_secs,
                "mse_v_target": err["mse"],
                "mse_per_frame": [round(x, 5) for x in err["per_frame"]],
                "mse_wrong_target_eps": wrong_eps,
                "mse_wrong_target_negx0": wrong_x0,
                "scalar_timestep_mse": scalar_mse,
                "scalar_vs_pertoken_rel_l2": rel_diff,
                "hidden_states": {
                    str(k): {"shape": list(v.shape), "mean": v.mean().item(), "std": v.std().item()}
                    for k, v in sorted(hs.pooled.items())
                },
            }
        )
        print(json.dumps({k: v for k, v in sweep[-1].items() if k != "hidden_states"}), flush=True)

    rec["sweep"] = sweep

    # ---- 4. risk #1: what happens without latent normalisation ----------------------
    z_t_raw, _, v_tgt_raw = W.flow_noise(z0_raw, args.timesteps[len(args.timesteps) // 2], seed=args.seed)
    ts_tok = W.per_token_timestep(z_t_raw, args.timesteps[len(args.timesteps) // 2])
    v_pred_raw = W.forward_v(tf, z_t_raw, ts_tok, pe)
    rec["risk1_no_normalisation"] = {
        "t": args.timesteps[len(args.timesteps) // 2],
        "mse_v_target_unnormalised": ((v_pred_raw - v_tgt_raw) ** 2).mean().item(),
        "mse_v_target_normalised": sweep[len(args.timesteps) // 2]["mse_v_target"],
    }

    # ---- 5. seed sensitivity teaser (full treatment is T3/E0) ------------------------
    t_mid = args.timesteps[len(args.timesteps) // 2]
    seeds = []
    for s in range(3):
        z_t, _, v_tgt = W.flow_noise(z0, t_mid, seed=s)
        ts_tok = W.per_token_timestep(z_t, t_mid)
        v = W.forward_v(tf, z_t, ts_tok, pe)
        seeds.append(((v - v_tgt) ** 2).mean().item())
    rec["seed_teaser"] = {"t": t_mid, "mses": seeds,
                          "spread": max(seeds) - min(seeds)}

    rec["vram_final"] = W.vram_report("final")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rec, indent=2, ensure_ascii=False))
    print(f"\nwrote {args.out}")
    print(json.dumps({"model": rec["model"], "vram": rec["vram_final"],
                      "risk1": rec["risk1_no_normalisation"], "seed_teaser": rec["seed_teaser"]}, indent=2))


if __name__ == "__main__":
    import os

    HUB = str(Path(os.environ["HF_HOME"]) / "hub")
    main()
