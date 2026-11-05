"""Build results/E17_moe_experts/manifest.json from what is on disk."""
import glob, hashlib, json, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E17_moe_experts"
CACHE = ROOT / "cache"

def sha(p):
    with open(p,"rb") as f: return hashlib.sha256(f.read()).hexdigest()[:16]
def n_pt(d): return len(list((CACHE/d).glob("*.pt")))
def ver(c):
    try: return subprocess.run(c,shell=True,capture_output=True,text=True,timeout=120).stdout.strip()
    except Exception: return "unknown"

dirs=[f"wan22a14b_{e}_feats_{w}_intphys2{s}" for e in ("low","high")
      for w in ("real","random") for s in ("","_cj")]
m={
 "task":"E17: two experts of one MoE package — does the noise band an expert specialises in change D1/D2?",
 "created": time.strftime("%Y-%m-%d %H:%M:%S"),
 "model":{"repo":"Wan-AI/Wan2.2-T2V-A14B-Diffusers",
          "revision":"5be7df9619b54f4e2667b2755bc6a756675b5cd7",
          "experts":{"high":{"subfolder":"transformer","params_b":14.0,"layers":40,
                             "noise_band":"sigma >= boundary_ratio (0.875)"},
                     "low":{"subfolder":"transformer_2","params_b":14.0,"layers":40,
                            "noise_band":"sigma < boundary_ratio (0.875)"}},
          "boundary_ratio":0.875,
          "boundary_source":"model_index.json, verified on disk"},
 "geometry":{"height":512,"width":512,"frames":49,"latent_shape":[1,16,13,64,64],
             "tokens":13312,"patch":[1,2,2],
             "vae":"AutoencoderKLWan, z_dim 16 (same as Wan2.1, so no per-family normalisation change)"},
 "in_range_at_t600":{"low":True,"high":False,
                     "why":"t=600 => sigma 0.600 < 0.875, so the LOW-noise expert is the in-range one; the high-noise expert is run outside the band it was trained on (E17_PREREG.md §4.1)"},
 "data":{"dataset":"IntPhys2 Main","n_plain":1012,"n_cj":506},
 "extraction_counts":{d:n_pt(d) for d in dirs},
 "probe":{"folds":5,"inner_folds":3,"C_grid":[1e-3,1e-2,1e-1,1.0,10.0],
          "grouping":"intphys2 scene_index","feature_seeds":[0,1,2],"threads":6,
          "threads_note":"the probe is not bit-reproducible across BLAS thread counts (E16); this run used 6, so its numbers are internally consistent but not bit-comparable to E8's 4-thread run at the 4th decimal"},
 "deviations":[
  {"what":"high-noise expert used at t=600 (out of range)",
   "why":"boundary_ratio 0.875 means it only runs above sigma 0.875",
   "consequence":"its AUC describes 'a 14B denoiser trained on another noise band', not its competence in its own regime"},
  {"what":"both experts share one random lower bound",
   "why":"identical architecture + identical init seed => bit-identical random weights (verified with torch.equal)",
   "consequence":"Q2's delta is real-side only: it is (delta real AUC)/(one shared denominator), so it carries less independent information than the pre-registration assumed"},
  {"what":"Q2 conflates specialisation with generalisation",
   "why":"neither expert is measured inside its own noise band at the same t",
   "consequence":"a null result here means 'no detectable difference between the experts as tested', NOT 'noise-band specialisation does not affect readability'"}],
 "env":{"python":ver("python3 -V"),
        "torch":ver('python3 -c "import torch;print(torch.__version__)" 2>/dev/null'),
        "diffusers":ver('python3 -c "import diffusers;print(diffusers.__version__)" 2>/dev/null'),
        "gpu":"4x NVIDIA B200 (183 GB each)"},
 "scripts":{p:sha(p) for p in sorted(glob.glob("scripts/e17_*.py"))},
 "timing":{"s_per_clip":"2.65-2.91","peak_vram_gb":"31.6 (real) / 54.0 (random)","wall_hours_4gpu":4.7},
 "deliverables":["E17_PREREG.md","E17_cells.json","E17_verdict.json","E17_REPORT.md","manifest.json"],
}
(OUT/"manifest.json").write_text(json.dumps(m,indent=1))
print(json.dumps(m["extraction_counts"],indent=1)); print("wrote",OUT/"manifest.json")
