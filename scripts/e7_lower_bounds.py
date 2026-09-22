"""E7 addendum: the VAE-latent and raw-pixel lower bounds on IntPhys 2.

Motivation comes from E6.  The Invisible Hand paper (2606.05328) states in its abstract:

    "this signal is absent from the VAE latent input and emerges inside the denoising
     transformer itself"

Our LikePhys number contradicts that (vae_latent probe AUC 0.8200), but PREREG §5 forbids
reading that as a direct contradiction because the settings differ: they probe IntPhys /
InfLevel with per-video accuracy, we probed LikePhys with AUC.

IntPhys 2 moves our setting materially closer to theirs (same benchmark family, same
violation-of-expectation construction), so running the same VAE-latent probe here gives a
comparison that is still not identical but is far less confounded.  The features are
already cached in every record, so this costs nothing extra to extract.

Still not a like-for-like replication: IntPhys 2 is the successor benchmark, not IntPhys,
and the metric is AUC not per-video accuracy.  Reported as "closer, still not equal".

Run: python scripts/e7_lower_bounds.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e7_analyse import SEEDS, cache_dir, probe

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E7_intphys2"
T = "600"


def load_simple(backbone: str, field: str):
    """vae_latent_pooled [13,48] or raw_pixel_32 [13,3072], flattened."""
    d = cache_dir(backbone, "plain")
    rows, meta = [], []
    for f in sorted(d.glob("*.pt")):
        rec = torch.load(f, map_location="cpu", weights_only=False)
        rows.append(rec[field].reshape(-1).numpy().astype(np.float32))
        c = rec["clip"]
        meta.append({"label": c["label"], "scene_index": c["scene_index"],
                     "condition": c["condition"], "camera": c["camera"]})
    return np.stack(rows), meta


def main():
    from threadpoolctl import threadpool_limits

    res = {"note": ("VAE-latent / raw-pixel lower bounds on IntPhys 2, prompted by the "
                    "Invisible Hand claim that the signal is absent from the VAE latent"),
           "sources": {}}

    with threadpool_limits(limits=8):
        for field, name in (("vae_latent_pooled", "vae_latent"),
                            ("raw_pixel_32", "raw_pixel")):
            X, meta = load_simple("real", field)
            y = np.array([m["label"] for m in meta])
            g = np.array([m["scene_index"] for m in meta])
            per = [probe(X, y, g, seed=7000 + s) for s in SEEDS]
            A = np.array([p[0] for p in per], float)
            D = np.array([p[1] for p in per], float)
            res["sources"][name] = {
                "dim": int(X.shape[1]), "n": int(len(y)),
                "auc_mean": float(np.nanmean(A)),
                "auc_seed_spread": float(np.nanstd(A, ddof=1)),
                "dprime_mean": float(np.nanmean(D)),
                "per_seed_auc": [float(x) for x in A],
            }
            print(f"  {name:11s} dim={X.shape[1]:5d} n={len(y)} "
                  f"AUC {np.nanmean(A):.4f}±{np.nanstd(A, ddof=1):.4f}  "
                  f"d' {np.nanmean(D):6.3f}", flush=True)

    v = json.loads((OUT / "E7_verdict.json").read_text())
    dit = v["D1"]["rows"][0]["real_auc"]
    rnd = v["D1"]["rows"][0]["random_auc"]
    vae = res["sources"]["vae_latent"]["auc_mean"]
    pix = res["sources"]["raw_pixel"]["auc_mean"]

    res["comparison"] = {
        "real_dit_auc": dit, "random_dit_auc": rnd,
        "vae_latent_auc": vae, "raw_pixel_auc": pix,
        "likephys_vae_latent_auc": 0.8200,
        "invisible_hand_claim": ("\"this signal is absent from the VAE latent input and "
                                 "emerges inside the denoising transformer itself\" "
                                 "(2606.05328, abstract)"),
        "reading": (
            f"On IntPhys 2 the VAE-latent probe reaches AUC {vae:.4f} against {dit:.4f} for "
            f"the DiT states and {rnd:.4f} for the randomly initialised DiT. "
            + ("The VAE latent carries little of the signal here, which is consistent with "
               "the Invisible Hand claim in a setting much closer to theirs than LikePhys "
               "was -- and it suggests our LikePhys value of 0.8200 reflects something "
               "about LikePhys rather than a general property."
               if vae < dit - 0.05 else
               "The VAE latent carries a substantial part of the signal here too, so the "
               "'absent from the VAE latent' claim does not reproduce even in this closer "
               "setting.")
            + " Still not a like-for-like replication: IntPhys 2 is the successor of the "
              "benchmark they used, and we report AUC where they report per-video accuracy "
              "(PREREG §5)."),
    }
    (OUT / "E7_lower_bounds.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\n  {res['comparison']['reading']}")
    print(f"\nwrote {OUT/'E7_lower_bounds.json'}")


if __name__ == "__main__":
    main()
