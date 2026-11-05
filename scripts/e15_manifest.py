"""Build results/E15_likephys/manifest.json from what is on disk."""
import glob, hashlib, json, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "E15_likephys"
CACHE = ROOT / "cache"

def sha(p):
    with open(p,"rb") as f: return hashlib.sha256(f.read()).hexdigest()[:16]
def n_pt(d): return len(list((CACHE/d).glob("*.pt")))
def ver(c):
    try: return subprocess.run(c,shell=True,capture_output=True,text=True,timeout=120).stdout.strip()
    except Exception: return "unknown"

dirs=["vjepa_feats_real_likephys","vjepa_feats_random_likephys",
      "hunyuan_feats_real_likephys","hunyuan_feats_random_likephys"]
m={
 "task":"E15: LikePhys half of the cross-paradigm D1 test (unfreezes E13 §8 / E14 §5)",
 "created": time.strftime("%Y-%m-%d %H:%M:%S"),
 "models": {
   "vjepa2_vitl":{"repo":"facebook/vjepa2-vitl-fpc64-256",
     "revision":"b3c1679b7c34d3255ef3547f27c7b226aefab26f","params_b":0.326,
     "layers":24,"geometry":{"h":256,"w":256,"frames":64,"tokens":8192},
     "non_generative":True},
   "hunyuan":{"repo":"hunyuanvideo-community/HunyuanVideo",
     "revision":"e8c2aaa66fe3742a32c11a6766aecbf07c56e773","params_b":12.8,
     "layers":60,"n_double":20,"n_single":40,
     "geometry":{"h":512,"w":512,"frames":49,"tokens":13312},
     "non_generative":False}},
 "data":{"dataset":"LikePhys","n_physics_scored":750,"n_groups":120,
         "manifest":"results/T3_null_floor/manifest.json",
         "label_filter":"clip['label'] in (0,1) — the temporal (120) and appearance (50) "
                        "groups carry label=None by design and are controls, not physics pairs",
         "grouping":"(scenario, subgroup)"},
 "extraction_counts":{d:n_pt(d) for d in dirs},
 "deviations":[
  {"what":"LikePhys clips are 56-61 frames but V-JEPA 2 requires 64",
   "why":"the position encoding is fixed at 64 frames",
   "consequence":"read_spanning repeats 1-8 indices (60-frame clip: 4 duplicated of 64), so "
                 "roughly 2 of the 32 temporal positions are frozen. Applied identically to "
                 "the trained and random runs, so the contrast is unaffected; stated because "
                 "it makes the temporal axis slightly less smooth than on IntPhys 2"},
  {"what":"probe seed not fixed by the pre-registration for V-JEPA 2",
   "why":"E15_PREREG.md assumed one probe seed per cell, which holds for the diffusion "
         "backbones (feature seeds 0/1/2 give the spread) but not for a deterministic encoder",
   "consequence":"its D1 straddles 50% across probe seeds (mean 49.41%, sd 2.29%, 5/6 below); "
                 "reported as `at_threshold`, not as a pass chosen by seed"},
  {"what":"D2 not re-run on LikePhys for the two new models",
   "why":"out of scope for the single question E15 asked",
   "consequence":"LikePhys D2 evidence remains E8's 6 generative models; the 50 appearance "
                 "clips are already in the extracted 920, so adding it is cheap"}],
 "env":{"python":ver("python3 -V"),
        "torch":ver('python3 -c "import torch;print(torch.__version__)" 2>/dev/null'),
        "gpu":"4x NVIDIA B200 (183 GB each)"},
 "scripts":{p:sha(p) for p in sorted(glob.glob("scripts/e15_*.py"))},
 "timing":{"vjepa_s_per_clip":0.24,"hunyuan_s_per_clip":3.05,"wall_min_4gpu":50},
}
(OUT/"manifest.json").write_text(json.dumps(m,indent=1))
print(json.dumps(m["extraction_counts"],indent=1)); print("wrote",OUT/"manifest.json")
