"""Audit: every decimal number in a report must be traceable to that task's JSON.

This is the machine-checkable half of the project's fourth red line ("no number without an
owner"). It is deliberately naive — it matches every `0.xxx` / `0.xxxx` token in the report
and asks whether that value appears anywhere in the task's JSON files — because a cleverer
matcher (e.g. only numbers inside tables) would miss the numbers that matter most: the ones
quoted inline in prose, where nobody looks.

Two categories are legitimately NOT in the JSON and are allow-listed per report, with the
reason recorded here rather than silently skipped:

  * values quoted from an EARLIER deliverable (e.g. E8's 0.5506 appearing in E12 as the
    baseline it reproduced), and
  * values quoted from a SUPERSEDED draft of the same report, written up specifically so the
    correction is visible (E13 §3.1 quotes its own wrong 0.032).

Both are references, not claims. Anything else missing is a real defect.

Run: python scripts/audit_reports.py
Exit code 1 if any report has an untraceable number.
"""

import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# E8_REPORT.md is audited only for the section appended by the later tasks (§16), because
# the 2026-10-24 body predates this audit and carries four values that are genuinely not in
# any JSON. Those are registered as a known defect rather than edited away — the project
# forbids hand-editing delivered numbers, and one of them (0.4483, the §13 LTX gap) is the
# already-documented §13-vs-§14 contradiction in E8_HANDOFF.md §5.2. See KNOWN_E8_DEFECTS.
REPORTS = {
    "results/E12_token_budget/E12_REPORT.md": ("results/E12_token_budget", None),
    "results/E13_non_generative/E13_REPORT.md": ("results/E13_non_generative", None),
    "results/E14_hunyuan/E14_REPORT.md": ("results/E14_hunyuan", None),
    "results/E15_likephys/E15_REPORT.md": ("results/E15_likephys", None),
    "results/E16_probe_seed/E16_REPORT.md": ("results/E16_probe_seed", None),
    "results/E17_moe_experts/E17_REPORT.md": ("results/E17_moe_experts", None),
    "results/E9_saturation/E9_REPORT.md": ("results/E9_saturation", None),
    "results/E11_readout/E11_REPORT.md": ("results/E11_readout", None),
    "results/E18_own_band/E18_REPORT.md": ("results/E18_own_band", None),
    # E19 quotes E8's six-model lower-bound table to establish what "reproduces" means,
    # so it is audited against results/ as a whole -- the same rule the round docs follow.
    "results/E19_h3_scale/E19_REPORT.md": ("results", None),
    # E20 quotes E8's delivered D1 values beside its own intervals, so it is audited against
    # results/ as a whole for the same reason E19 is.
    "results/E20_bootstrap_ci/E20_REPORT.md": ("results", None),
    # E21 quotes E8's IntPhys 2 reference AUC (0.6458) as its comparison baseline.
    "results/E21_physics_iq_real/E21_REPORT.md": ("results", None),
    # The round summary quotes numbers from every task, so it is audited against results/ as a
    # whole rather than one directory — that is the point of a cross-task document.
    "results/ROUND_2026-10-28_E12_E17.md": ("results", None),
    "results/ROUND_2026-10-29_E9_E11.md": ("results", None),
    "results/ROUND_2026-10-31_E18_H3.md": ("results", None),
    "results/ROUND_2026-11-01_E19.md": ("results", None),
    "results/E8_second_backbone/E8_REPORT.md": ("results", "## 16."),
}

# Pre-existing E8 values with no JSON owner. NOT fixed here (red line: no hand-editing
# delivered numbers); they are listed so the defect is tracked rather than silently passing.
KNOWN_E8_DEFECTS = {
    "0.9768": "§3.1 sigma_eff at t=600 on the VP schedule — computed ad hoc, never stored",
    "0.4449": "§3.1 SNR of the t=298 match — computed ad hoc, never stored",
    "0.4483": "§13 LTX gap, superseded by §14 (E8_HANDOFF.md §5.2 already flags this)",
    "0.5478": "§15 layer scan value, only exists in a log (E8_HANDOFF.md §5.3 records this)",
}

# (report, value) -> why it is not in that task's own JSON
ALLOWED = {
    ("E13_REPORT.md", "0.032"): "quotes E13's own superseded draft; disclosed in §3.1",
    ("E13_REPORT.md", "0.0000"): "quotes E8's CogVideoX collapse value, cross-task reference",
    ("E8_REPORT.md", "0.5506"): "E12 baseline reproduced from E8, quoted back into E8 §16",
    ("E15_REPORT.md", "0.2759"): "E13's IntPhys2 V-JEPA gap, cited for the cross-dataset contrast",
    ("E15_REPORT.md", "0.276"): "same value as 0.2759, rounded in the same sentence",
    ("E15_REPORT.md", "0.9641"): "E8's LikePhys appearance AUC range (E8 §14.2), cited as context",
    # --- cross-task D1 percentages, quoted for comparison against E8's published table ---
    ("E13_REPORT.md", "79.4%"): "E8's Mochi IntPhys2 D1, quoted for the range 79.4-99.1",
    ("E13_REPORT.md", "91.2%"): "E8's Wan2.1-1.3B IntPhys2 D1, same range",
    ("E13_REPORT.md", "85.3%"): "E8's CogVideoX-2b IntPhys2 D1, same range",
    ("E13_REPORT.md", "92.5%"): "E8's CogVideoX-2b SNR-matched t=298 cell, cited as sensitivity",
    ("E13_REPORT.md", "97.7%"): "share of noise at t=600 on the VP schedule (E8 §3.1)",
    ("E13_REPORT.md", "43.5%"): "E8's CogVideoX-5b LikePhys D1, upper end of the generative band",
    ("E13_REPORT.md", "49.4%"): "E15's V-JEPA LikePhys D1, cited forward to the later task",
    ("E13_REPORT.md", "53.3%"): "relative depth 16/30, a definition not a measurement",
    ("E14_REPORT.md", "26.7%"): "relative depth 16/60, a definition not a measurement",
    ("E14_REPORT.md", "99.1%"): "E8's Wan2.1-14B IntPhys2 D1, top of the range",
    ("E15_REPORT.md", "28.5%"): "E8's Wan2.2-TI2V-5B LikePhys D1 (E7 reference row)",
    # --- E9 / E11 ---
    ("E9_REPORT.md", "0.500"): "the 0.500 the archived proposal recorded for shadow_camera; a "
                               "cross-document reference (archive/PROPOSAL_中文版.md:47), quoted "
                               "to show E9's 0.3662 is the same anomaly, worse",
    ("E11_REPORT.md", "0.8776"): "EXPERIMENTS.md §5.3.2's old MLP AUC, quoted as the flawed "
                                 "baseline E11 was built to replace",
    ("E11_REPORT.md", "0.9686"): "same section: the old linear AUC the MLP was compared against",
    ("E11_REPORT.md", "43.9%"): "same section: the old MLP training share (28.5% -> 43.9%)",
    ("E11_REPORT.md", "30.6%"): "the value this script's own first version produced when it "
                                "re-implemented the linear probe instead of importing it; quoted "
                                "as the reason that mistake is now forbidden in the code",
    ("E17_REPORT.md", "2.51 pp"): "E16's probe-partition sd median, cited as the Q2 threshold",
    ("E17_REPORT.md", "99.2501%"): "the 6-thread superseded run, kept in "
                                   "E17_cells.threads6_superseded.json as evidence",
    ("E17_REPORT.md", "99.1995%"): "same, 6-thread high-expert value",
    ("E17_REPORT.md", "0.0506"): "the 6-thread L16 delta, named explicitly in the §5.1 "
                                 "superseded-run table",
    ("E17_REPORT.md", "0.552"): "upper end of E12's LTX real-AUC ladder, cross-task reference",
    ("E17_REPORT.md", "0.600"): "sigma at t=600; derived from t/1000 and stated in E17's own "
                                "manifest under in_range_at_t600.why",
    ("E17_REPORT.md", "0.875"): "boundary_ratio, stored in E17 manifest model.boundary_ratio",
    ("E17_REPORT.md", "0.0326"): "the 6-thread Lrel delta, named in the §5.1 superseded table",
    ("ROUND_2026-10-28_E12_E17.md", "0.5212671690375497"): "the 4-thread probe value quoted from "
                                                          "E16's thread table",
    ("ROUND_2026-10-28_E12_E17.md", "0.5213261335383826"): "the 1-thread value, same table",
    ("ROUND_2026-10-28_E12_E17.md", "0.520435"): "the 3-thread value, same table",
    ("ROUND_2026-10-28_E12_E17.md", "0.5506"): "E12's 1792-token rung, in E12_cells_ltx.json",
    ("ROUND_2026-10-28_E12_E17.md", "0.5480"): "E12's 3328-token rung",
    ("ROUND_2026-10-28_E12_E17.md", "0.5519"): "E12's 6400-token rung",
    ("ROUND_2026-10-28_E12_E17.md", "0.5304"): "E12's 9472-token rung",
    ("ROUND_2026-10-28_E12_E17.md", "0.5397"): "E12's 11008-token rung",
    ("ROUND_2026-10-28_E12_E17.md", "0.5308"): "E12's 12800-token rung",
    ("ROUND_2026-10-28_E12_E17.md", "0.4483"): "cited as an example of E8's known defect, "
                                               "not as a live claim (see KNOWN_E8_DEFECTS)",
    ("E15_REPORT.md", "54.1%"): "ViT-H LikePhys D1 mean, in "
                                "E13_cells_vith_likephys.json (a different task's JSON)",
}


def collect(dirpath: str) -> set:
    pool = set()

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, bool) or o is None or isinstance(o, str):
            return
        elif isinstance(o, (int, float)):
            v = abs(float(o))
            for nd in (2, 3, 4, 5, 6, 7):
                pool.add(f"{v:.{nd}f}")

    for p in glob.glob(f"{dirpath}/**/*.json", recursive=True):
        try:
            walk(json.loads(Path(p).read_text()))
        except Exception as exc:
            print(f"  UNREADABLE {p}: {exc}")
    return pool


def main() -> int:
    bad = 0
    for rep, (d, section) in REPORTS.items():
        rp = ROOT / rep
        if not rp.exists():
            print(f"SKIP     {rep} (absent)")
            continue
        pool = collect(str(ROOT / d))
        text = rp.read_text()
        if section:                      # audit only the appended section
            i = text.find(section)
            if i < 0:
                print(f"SKIP     {rep} (section {section!r} absent)")
                continue
            text = text[i:]
        # Two formats count as a quoted number:
        #   0.xxx / 0.xxxx  (AUC-style, 3-4 decimals)
        #   xx.x%           (D1-style percentages, 1 decimal) — E16 and the verdict reports
        #                    quote D1 this way, and without this pattern the audit silently
        #                    examined a single number in a 200-line report.
        nums = set(re.findall(r"\b0\.\d{3,4}\b", text))
        pcts = set(re.findall(r"\b(\d{1,3}\.\d)%", text))
        name = Path(rep).name
        # A quoted percentage is usually the ROUNDED form of a stored ratio (0.928581 -> 92.9%),
        # so exact string matching is wrong here. Accept it if any stored value, scaled to
        # percent, rounds to the quoted figure.
        pct_nums = [float(f"{v}") for v in
                    {x for x in re.findall(r"[-+]?\d+\.\d+", " ".join(pool))}]

        def pct_in_pool(v: str) -> bool:
            """Stored either as a ratio (0.473 -> 47.3%) or already as a percent (47.299 -> 47.3%)."""
            want = float(v)
            return any(abs(round(x * 100, 1) - want) < 1e-9
                       or abs(round(x, 1) - want) < 1e-9 for x in pct_nums)

        missing = sorted([n for n in nums
                          if n not in pool and (name, n) not in ALLOWED]
                         + [f"{v}%" for v in sorted(pcts)
                            if not pct_in_pool(v) and (name, f"{v}%") not in ALLOWED])
        if missing:
            bad += 1
        print(f"{'OK' if not missing else 'MISSING':8s} {name:20s} {len(nums):3d} numbers, "
              f"{len([1 for _ in glob.glob(f'{d}/**/*.json', recursive=True)])} jsons"
              + (f"  -> {missing}" if missing else ""))

    allowed_hits = [(k, v) for k, v in ALLOWED.items()
                    if (ROOT / next(r for r, dd in REPORTS.items()
                                    if Path(r).name == k[0])).exists()]
    print(f"\n{len(ALLOWED)} allow-listed cross-task/draft quotes; "
          f"{len(allowed_hits)} of their reports present")
    print("reports with untraceable numbers:", bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
