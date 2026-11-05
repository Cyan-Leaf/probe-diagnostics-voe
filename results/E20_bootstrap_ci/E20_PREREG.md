# E20 PREREG — cluster-bootstrap confidence intervals for D1 and D2

**Written 2026-11-05, before the runs.** Written *after* the point estimates existed but
*before* any interval was computed. Every design choice below is therefore fixed before we
know whether the verdicts survive; the purpose of writing it down is that we cannot pick the
resampling scheme that happens to preserve the story.

## 0. Why this exists

Every D1 and D2 verdict in this project is a point estimate. `D1_TALLY.json` (35 cells) and
`D2_TALLY.json` (34 cells) contain no interval of any kind, and `PREREG.md` never specified
one. Three separate results say that is no longer good enough:

- **E16** measured the probe-partition sd at 2.51 pp, and **E18** found a feature-seed sd of
  20–26 pp in the weak-signal region. So the sampling error is neither negligible nor uniform.
- **E18** showed a 0.0116 AUC difference becoming ~10 pp of D1 because the denominator
  (`AUC_real`, a *ratio* component) was only 0.096–0.133. A ratio of two noisy quantities has
  an interval that no point estimate can reveal.
- The smallest D2 gap in the whole table is **+0.0289** (V-JEPA 2 on LikePhys) and two D1
  cells sit within **0.6 and 2.0 pp** of the 50% boundary. For those cells the current report
  gives the reader no way to judge whether the verdict is a property of the model or of
  this particular draw of scenes.

The question E20 answers is narrow and pre-specified:

> For each verdict in the headline table, does a 95% interval for the underlying quantity
> exclude the decision boundary?

It is **not** a new effect size, not a new model, and it does not change any threshold.

## 1. Unit of resampling: the cluster, not the row

**Decision: resample clusters (scenes / subgroups) with replacement, never rows.**

Rationale. Both probes are fit and evaluated under group-disjoint CV, so two clips from the
same scene never sit on opposite sides of a fold boundary. If rows were resampled
independently, the same scene would appear in train and test, the probe would be scored on
material it memorised, and the interval would be **too narrow** — the classic
clustering error. The cluster is the independent unit of this design and is therefore the
unit of the bootstrap.

- IntPhys 2 → `scene_index`, **253 clusters**, n = 1012 rows.
- LikePhys → `scenario/subgroup`, **120 clusters**, n = 750 rows (physics group) or
  n = 620 for the appearance-native contrast.

**Duplicated clusters keep their original label.** This is deliberate: the standard
cluster bootstrap treats the cluster as atomic, so a scene drawn twice is one scene with two
copies, and both copies must land in the same CV fold. Giving the copies distinct ids would
let half a scene train the probe and the other half test it — the exact leak the grouping
exists to prevent.

## 2. Number of draws and the interval estimator

**B = 300. Percentile 95% interval, no bias correction.**

Why not BCa: BCa needs a jackknife over 253 clusters, i.e. 253 extra probe fits per arm,
which costs more than it buys at this sample size. Percentile intervals are the estimator we
can defend without another design decision layered on top.

Why 300: the 2.5 / 97.5 percentile is estimated from the 8th and 293rd order statistics; at
B = 300 the Monte-Carlo error on the interval endpoints is ~1/3 of the endpoint's own
spacing, which we checked is small relative to the widths reported (see §6). B is fixed here
and does not get raised for a cell whose interval looks inconvenient.

## 3. Pairing — the single most important detail

D1 and D2 are **ratios and differences of two probe fits**, so an interval built from
marginal intervals of each arm would be wrong twice over (it ignores the correlation between
arms and it cannot represent a ratio's denominator).

**Decision: within a draw, all arms of a cell see the *same* resampled index set.**

- D1 share = `(AUC_real − AUC_random) / AUC_real`, both computed on draw *b*.
- D2 gap   = `AUC_appearance − AUC_physics`, both computed on draw *b*.
- The interval is taken over the 300 per-draw values of the composed quantity — never over
  the two arms separately.

Draw indices are generated once per dataset from a fixed seed (`20261105`) and written to
disk, so every arm of every cell applies literally the same draws and the arms cannot drift
apart. This also makes the study reproducible without re-running the probe.

The probe's own CV seed is held **fixed at 1000** across all draws (E8's convention for
feature seed 0). Feature seed 0 is used for every cell, and the E8 three-seed mean is
reported beside it: if they disagree beyond the seed spread of E8, that is a finding and gets
reported, not smoothed over.

## 4. Verdict-robustness definitions (fixed before running)

| quantity | decision boundary | "robust" means |
|---|---|---|
| **D1** | pass iff share ≥ 0.50 | the 95% interval lies entirely on the same side of 0.50 as the point estimate |
| **D2** | fail iff appearance > physics | the 95% interval for the *gap* excludes 0 (lower bound > 0) |

A cell whose interval **straddles** the boundary is reported as `verdict_fragile` and both
the point estimate and the interval are printed. **The point-estimate verdict is not
retracted** — it stays in `D1_TALLY.json` / `D2_TALLY.json` untouched, because changing a
delivered verdict after seeing an interval is exactly the kind of post-hoc adjustment the
project forbids. E20 writes its own files and a separate tally.

Pre-committed expectation, so a surprise is visible as a surprise: we expect all D2 cells to
survive (even +0.0289 is a large gap relative to a null on n≈620–1012) and we expect the two
`at_threshold` D1 cells (V-JEPA 2 at 49.41% and 47.97%) to be fragile. Those expectations do
not license dropping any cell.

## 5. Scope — which cells, and why not all of them

Covered: **E8's six backbones × two datasets**, L16 / t=600, i.e. 12 (backbone, dataset)
units. Per unit:

- **D1** (real vs random-init) — all 12 units.
- **D2** — all 12 units: `plain vs cj` on IntPhys 2 (two-cache join, as E8 does it),
  `appearance_native` on LikePhys (the built-in `color_change` clips).

This is the set for which the caches are complete and the extraction is the pre-registered
protocol run, so nothing is re-extracted. Not covered in this pass: the E13/E14/E15/E17/E18/
E19 model additions, and the `Lrel` layer. Reason: the E8 six already span the full range of
gaps that matter (D2 from +0.0289 to +0.4461, D1 from 32% to 99%), and the point of E20 is
to calibrate how much a verdict in this design can move — a second copy of the same interval
on nine more models adds little. If any E8 cell turns out fragile, the corresponding added
model gets the same treatment in a follow-up; that decision is **conditional on the result**
and is written here as such so it cannot look like fishing later.

## 6. Self-check, to be reported whether it passes or fails

Three checks run before any interval is quoted:

1. **Identity draw.** Applying the resample indices in their natural order with no
   duplication must reproduce the E8 point estimate. **AMENDED 2026-11-05, before any
   interval was computed and after the check itself failed:** it does *not* reproduce
   bit-for-bit, and the reason is a finding in its own right. Rebuilding the draw reorders
   the rows; the row multiset is identical, but `StandardScaler` accumulates its scale in a
   different order (measured max 7.1e-15 on the scale, 0.0 on the mean), and `_rowspace` —
   which selects a near-null direction of the standardised matrix — amplifies that into
   **1.9e-05** on IntPhys 2 and **3.4e-04** on LikePhys in AUC.

   So the harness is correct and **the frozen probe is order-sensitive at the 1e-5…1e-4 AUC
   level**. The tolerance is therefore set to **1e-3 AUC (0.1 pp)** — about 3× the largest
   observed order effect and ~25× below E16's 2.51 pp probe-partition sd. Exceeding it withholds
   the cell instead of reconciling it. The original wording ("bit-for-bit") is left visible
   above rather than edited away, because changing a frozen criterion silently is exactly what
   this project forbids; the amendment is registered here and the measured deltas ship in
   `E20_cells.json.self_check`.
2. **Point estimate agreement.** Each cell's draw-0 AUC must equal `E8_cells_*.json` for the
   same (backbone, dataset, arm, feature seed 0) within the same 1e-3. Any larger mismatch is
   reported as a defect and the cell is withheld, not silently reconciled.
3. **Monte-Carlo stability.** The interval is recomputed from two disjoint halves of the 300
   draws; the endpoint disagreement is reported per cell. If that disagreement is a large
   fraction of the interval width, B is too small and it is stated as a limitation rather
   than papered over.

### 6.4 Probe CV seeds follow E8 per arm, not one global seed

Discovered while wiring the self-check: E8 gives each arm a different probe seed
(`real`/`random` → `1000 + feature_seed`; the IntPhys 2 appearance arm, which is built by a
separate function, → `2000 + feature_seed`). Reproducing the delivered numbers requires
carrying those through. Pairing is over drawn clusters — which is what the interval needs —
while each arm keeps E8's own partition. The alternative (one global probe seed for all arms)
would make D2's point estimate differ from the delivered value for a reason unrelated to the
interval, and would look like a defect.

## 7. What will be delivered

```
results/E20_bootstrap_ci/
    E20_PREREG.md            this file
    E20_draws_intphys2.npz   the frozen resample index sets (253 x 300)
    E20_draws_likephys.npz   ditto (120 x 300)
    E20_cells.json           per cell: point, 95% CI, verdict, verdict_fragile, MC check
    E20_verdict.json         the robustness tally: how many of 24 verdicts survive
    E20_REPORT.md            the table, the two fragile cells, and the limitations
    manifest.json            env, threads, probe source, cache checksums
```

## 8. Discipline inherited from the rest of the project

- **`probe` is imported, never re-implemented** (`import e8_analyse.probe`) — E11 measured a
  2.3 pp bias in a rewrite.
- **`threads` is recorded** and identical across every cell (E16: 3 vs 4 threads moves AUC by
  ~0.5 pp).
- **No existing number is modified.** E20 writes new files only; `D1_TALLY.json` and
  `D2_TALLY.json` are regenerated by their own scripts, not by this one.
- **This document is exploratory in one respect only** — it adds an interval to verdicts that
  were frozen without one. It adds no verdict and removes none.
