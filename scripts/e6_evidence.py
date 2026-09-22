"""E6: harvest the evidence needed to fill the D1-D6 x paper audit table.

Hard rule 5: the table may only be filled from the papers' own text.  This script pulls
the passages that bear on each diagnostic so the judgement is made against quoted text
rather than memory, and so every cell can carry a locator.

It does NOT decide anything -- it prints candidate evidence for each (paper, diagnostic).
The verdicts are written by hand into results/E6_literature/E6_TABLE.md.

Run: python scripts/e6_evidence.py            # all papers, all diagnostics
     python scripts/e6_evidence.py --paper 2606.05328 --diag D1
"""

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "E6_literature" / "raw"

PAPERS = {
    "2606.05328": "Invisible Hand of Physics",
    "2603.14294": "Seeking Physics in Diffusion Noise",
    "2606.09646": "Punzo et al. (probing across pretraining paradigms)",
    "2510.11512": "LikePhys",
}

# Patterns chosen to catch the *method* words a paper would use if it did the check,
# not the concept words we happen to use for it.
DIAG = {
    "D1": {
        "name": "random-init lower bound",
        "pats": [r"random(ly)?[- ]initiali[sz]", r"untrained", r"from scratch",
                 r"random(ly)? initiali[sz]ed (network|model|backbone|weights)",
                 r"random weights", r"random(ised)? (network|encoder)"],
    },
    "D2": {
        "name": "attribute-irrelevant perturbation control",
        "pats": [r"control (condition|experiment)", r"shuffl", r"colou?r (jitter|change|shift)",
                 r"appearance (cue|control|shortcut)", r"single[- ]frame",
                 r"static (cue|appearance)", r"low[- ]level (cue|statistic)"],
    },
    "D3": {
        "name": "ceiling / saturation check",
        "pats": [r"ceiling", r"saturat", r"near[- ]perfect", r"\b100% accuracy",
                 r"at chance", r"chance level"],
    },
    "D4": {
        "name": "noise floor",
        "pats": [r"random(ised|ized)? label", r"permut", r"null (distribution|model|hypothesis)",
                 r"chance (baseline|performance)", r"(three|3|multiple) (random )?seeds",
                 r"standard (deviation|error) (over|across)", r"error bar", r"confidence interval"],
    },
    "D5": {
        "name": "more than one readout / cross-readout consistency",
        "pats": [r"linear probe", r"MLP probe", r"attentive prob", r"likelihood",
                 r"denoising (error|loss|objective)", r"ELBO", r"reconstruction error",
                 r"two (different )?readout", r"both (metrics|readouts)"],
    },
    "D6": {
        "name": "bounded statistic in a saturated regime",
        "pats": [r"\bAUC\b", r"\bAUROC\b", r"accuracy", r"\bPPE\b", r"VoE|violation of expectation",
                 r"d'|d-prime|discriminability", r"effect size"],
    },
}


def load(pid):
    p = RAW / f"{pid}.txt"
    return p.read_text(errors="ignore") if p.exists() else None


def hits(text, pats, ctx, cap):
    seen, out = set(), []
    for pat in pats:
        for m in re.finditer(pat, text, re.I):
            a, b = max(0, m.start() - ctx), min(len(text), m.end() + ctx)
            key = a // 400
            if key in seen:
                continue
            seen.add(key)
            out.append({"pattern": pat, "pos": m.start(),
                        "text": re.sub(r"\s+", " ", text[a:b]).strip()})
            if len(out) >= cap:
                return out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", default=None)
    ap.add_argument("--diag", default=None)
    ap.add_argument("--ctx", type=int, default=320)
    ap.add_argument("--cap", type=int, default=5)
    ap.add_argument("--json-out", action="store_true")
    args = ap.parse_args()

    papers = [args.paper] if args.paper else list(PAPERS)
    diags = [args.diag] if args.diag else list(DIAG)

    store = {}
    for pid in papers:
        text = load(pid)
        if text is None:
            print(f"### {pid}: NO LOCAL TEXT -> every cell must read 无法核实")
            store[pid] = {"available": False}
            continue
        store[pid] = {"available": True, "chars": len(text), "diags": {}}
        print(f"\n{'='*90}\n### {pid}  {PAPERS[pid]}   ({len(text)} chars)\n{'='*90}")
        for d in diags:
            h = hits(text, DIAG[d]["pats"], args.ctx, args.cap)
            store[pid]["diags"][d] = h
            print(f"\n--- {d}: {DIAG[d]['name']}  ({len(h)} passage(s)) ---")
            if not h:
                print("    [no matching passage -> candidate 未报告]")
            for x in h:
                print(f"    @{x['pos']:6d} [{x['pattern']}]")
                print(f"      {x['text'][:400]}")

    if args.json_out:
        p = RAW.parent / "evidence.json"
        p.write_text(json.dumps(store, indent=2, ensure_ascii=False))
        print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
