"""Score the human baseline once the rater sheets come back.

Answers the question PROTOCOL_VALIDITY §5 poses: does human judgement agree with the
protocol's verdict about which units are trustworthy?  The decisive cell is "protocol says
untrustworthy AND humans cannot tell apart" -- the protocol confirmed by a source that
knows nothing about the model.

**Forced choice** (owner's decision, 2026-09-19).  There is no `cannot_tell` option: every
pair gets a left/right answer plus a mandatory 1-5 confidence, where 1 is labelled
"完全分不出，纯猜".  The reason is that letting raters skip hard pairs would bias accuracy
upward on exactly the pairs the claim is about -- people only commit when they see
something -- and that accuracy would then not be comparable to the probe's AUC, which is
itself a forced ranking.  The "cannot tell" signal is not lost: it is the rate of
confidence = 1, reported alongside accuracy.

Legacy `cannot_tell` rows (from the CSV-only version of the package) are still parsed and
counted separately, so a mixed batch does not silently drop rows.

Run: python scripts/t10_human_score.py results/T10_validity/human_eval/filled/*.csv
     ./humaneval/serve.sh score          # exports from the platform DB and runs this
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HE = ROOT / "results" / "T10_validity" / "human_eval"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sheets", nargs="+")
    ap.add_argument("--out", default=str(HE / "human_baseline_results.json"))
    args = ap.parse_args()

    key = {p["pair_id"]: p for p in json.loads((HE / "ANSWER_KEY.json").read_text())}

    per_pair = defaultdict(list)
    raters = []
    for s in args.sheets:
        name = Path(s).stem
        raters.append(name)
        with open(s) as f:
            for row in csv.DictReader(f):
                pid = (row.get("pair_id") or "").strip()
                if pid not in key:
                    continue
                choice = (row.get("your_choice(left/right/cannot_tell)") or "").strip().lower()
                if choice not in ("left", "right", "cannot_tell"):
                    continue
                try:
                    conf = int((row.get("confidence(1-5)") or "0").strip())
                except ValueError:
                    conf = 0
                per_pair[pid].append({
                    "rater": name, "choice": choice, "confidence": conf,
                    "same_look": str(row.get("same_look", "")).strip() in ("1", "true", "True"),
                })

    if not per_pair:
        raise SystemExit("no usable responses found -- check the sheet column names")

    rows = []
    for pid, resp in per_pair.items():
        k = key[pid]
        truth = k["_answer_violation_side"]
        decided = [r for r in resp if r["choice"] != "cannot_tell"]
        correct = [r for r in decided if r["choice"] == truth]
        # Forced choice: accuracy is over ALL responses, which is what makes it comparable
        # to the probe's AUC.  `frac_guess` (confidence == 1) is the subjective
        # "cannot tell" rate and must be read next to it.
        guesses = [r for r in resp if r["confidence"] == 1]
        rows.append({
            "pair_id": pid, "scenario": k["scenario"],
            "violation_kind": k["_violation_kind"],
            "saturated": k["_scenario_saturated"],
            "probe_real_auc": k["_probe_real_auc"],
            "probe_delta": k["_probe_delta"],
            "n_raters": len(resp),
            "n_legacy_cannot_tell": len(resp) - len(decided),
            "accuracy": (len(correct) / len(decided)) if decided else None,
            "frac_guess": len(guesses) / len(resp) if resp else None,
            "frac_same_look": (sum(1 for r in resp if r.get("same_look")) / len(resp)
                               if resp else None),
            "mean_confidence": float(np.mean([r["confidence"] for r in resp if r["confidence"]]))
                               if any(r["confidence"] for r in resp) else None,
        })

    def agg(sel):
        acc = [r["accuracy"] for r in sel if r["accuracy"] is not None]
        return {
            "n_pairs": len(sel),
            "human_accuracy": float(np.mean(acc)) if acc else None,
            "frac_guess": float(np.mean([r["frac_guess"] for r in sel
                                         if r["frac_guess"] is not None])),
            "frac_same_look": float(np.mean([r["frac_same_look"] for r in sel
                                             if r["frac_same_look"] is not None])),
            "mean_confidence": float(np.nanmean([r["mean_confidence"] or np.nan for r in sel])),
        }

    by_scen = {}
    for sc in sorted({r["scenario"] for r in rows}):
        by_scen[sc] = agg([r for r in rows if r["scenario"] == sc])
        by_scen[sc]["probe_real_auc"] = next(r["probe_real_auc"] for r in rows
                                            if r["scenario"] == sc)
        by_scen[sc]["saturated"] = next(r["saturated"] for r in rows if r["scenario"] == sc)

    # the 2x2 that PROTOCOL_VALIDITY §5 asks for
    cells = {"protocol_untrustworthy_humans_cannot": [],
             "protocol_untrustworthy_humans_can": [],
             "protocol_trustworthy_humans_can": [],
             "protocol_trustworthy_humans_cannot": []}
    for sc, v in by_scen.items():
        untrust = v["saturated"]
        humans_can = (v["human_accuracy"] or 0) >= 0.75
        k = (f"protocol_{'untrustworthy' if untrust else 'trustworthy'}_"
             f"humans_{'can' if humans_can else 'cannot'}")
        cells[k].append(sc)

    out = {
        "n_raters": len(raters), "raters": raters,
        "n_pairs_answered": len(rows),
        "overall": agg(rows),
        "saturated_units": agg([r for r in rows if r["saturated"]]),
        "non_saturated_units": agg([r for r in rows if not r["saturated"]]),
        "per_scenario": by_scen,
        "per_pair": rows,
        "agreement_cells": cells,
        "reading": (
            "The cell that confirms the protocol is "
            "'protocol_untrustworthy_humans_cannot'. Scenarios there are ones where the "
            "protocol said the probe number carries no resolution and humans independently "
            "could not tell the pair apart either. Scenarios in "
            "'protocol_untrustworthy_humans_can' are the warning sign: the protocol may be "
            "too strict there and should be re-examined."),
        "caveat": ("forced choice, so accuracy is over every response and is directly "
                   "comparable to the probe AUC. frac_guess (confidence = 1, labelled "
                   "完全分不出、纯猜) is the subjective 'cannot tell' rate and must be read "
                   "next to accuracy: accuracy near 0.5 WITH a high frac_guess is the "
                   "signature of a genuinely indiscriminable pair, whereas accuracy near "
                   "0.5 with high confidence would instead suggest the raters were "
                   "systematically misled."),
    }
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"{len(raters)} rater(s), {len(rows)} pairs")
    hdr = "{:16s} {:>7s} {:>4s} {:>10s} {:>10s} {:>10s} {:>6s}".format(
        "scenario", "probe", "sat", "human acc", "frac guess", "same look", "conf")
    print("\n" + hdr)
    for sc, v in by_scen.items():
        print("{:16s} {:7.3f} {:>4s} {:10.3f} {:10.2f} {:10.2f} {:6.2f}".format(
            sc, v["probe_real_auc"], str(v["saturated"]),
            v["human_accuracy"] if v["human_accuracy"] is not None else float("nan"),
            v["frac_guess"], v["frac_same_look"], v["mean_confidence"]))
    print("\n=== agreement cells ===")
    for k, v in cells.items():
        print(f"  {k:44s} {v or '-'}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
