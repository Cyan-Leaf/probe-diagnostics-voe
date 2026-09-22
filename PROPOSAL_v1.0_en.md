# What Do Physics Probes Actually Measure?
### A Diagnostic Protocol and Its Limits

**Course project proposal · Cognitive Reasoning · v1.0 · 2026-09-22**

---

## 1. Introduction

Take one video diffusion model, one set of clips, one linear classifier, one layer, one
denoising timestep. Change nothing except what the labels mean.

| Task on IntPhys 2 (same features, same folds) | Probe AUC |
|---|---|
| physically **possible** vs **impossible** | **0.6458** |
| physically possible vs the *same clips recoloured* (hue +0.35, physics untouched) | **1.0000** |

The recolouring is physically legal: every frame gets the identical HSV shift, so object
trajectories, contacts and occlusions are bit-identical to the original. We verified this
numerically — the V channel is unchanged to 0.000000 and the frame-to-frame difference
profile correlates at 0.9989 with the original. A 7× weaker recolour (mean RGB change
0.0425, barely visible) still reaches 0.9980.

**What this does and does not show.** It does **not** show that the model fails to
understand physics — this experiment says nothing about that. What it shows is narrower and,
we think, more useful: **the number 0.6458 cannot be used to support the claim "this
representation encodes physical plausibility."** The same probe on the same representation
produces a *higher* number for a manipulation that contains no physics at all. Whatever the
probe is reading, its magnitude is not evidence about physics.

This is a claim about **how conclusions are drawn**, not about what models know. Our object
of study is the inference step that turns a probe AUC into a statement about representations.

**Why this matters now.** Three recent papers reach mutually incompatible conclusions from
the same family of measurement:

| Paper | Conclusion |
|---|---|
| Invisible Hand (`2606.05328`) | physical plausibility is linearly decodable from DiT states, 81.27% |
| Punzo et al. (`2606.09646`) | diffusion representations are *weaker* than V-JEPA on the same task |
| LikePhys (`2510.11512`) | denoising likelihood ranks models by physical plausibility |

They differ in dataset, in how internal states are obtained, in labels, and in readout. Our
position is that the disagreement is not primarily about models — it is about the
**measurement**, and there is currently no agreed way to tell a real representational effect
from an artefact of how the benchmark was built.

**What we propose.** A diagnostic protocol of six checks (D1–D6) that a
representation-level physics claim should pass before it is believed, plus experimental
evidence about the protocol's own validity — including a case where it fails.

---

## 2. Related Work

**Probing methodology.** The checks we assemble are not new to probing as a field.
Hewitt and Liang (2019) introduced **control tasks**: construct a task only the probe itself
could fit, and require *selectivity* (high on the target, low on the control). Zhang and
Bowman (2018) observed that probes on **randomly initialised** representations perform
surprisingly well; Voita and Titov (2020) restated this and showed that *differences in probe
accuracy* "do not substantially favour pretrained representations over randomly initialized
ones", proposing **MDL codelength** instead. Elazar et al. (2020) argued that probing results
do not license behavioural conclusions, and that what matters is how information is *used*.

> On Voita and Titov: their contribution is the *remedy* (codelength), not only the critique.
> Since our D1 is an accuracy-family statistic — precisely what they argue against — we also
> report the online prequential code (§4.1). Zhang and Bowman (2018) we cite **as relayed by**
> Voita and Titov; we have not obtained the original.

**Video physics evaluation.** IntPhys 2 (`2506.09849`) and LikePhys (`2510.11512`) are
violation-of-expectation benchmarks; V-JEPA-style encoders are the usual comparison point.
`2609.04264` reports "physical representation laziness" in JEPA-style latent world models —
independent, same-direction evidence from a different model class and method.

**The gap.** We audited the four papers above against D1–D6 using mechanical term search on
their full texts. **None reports a random-initialisation lower bound**: hits for `untrained`
and `from scratch` are zero in all four. Two near-misses are not the same check —
`2603.14294`'s "Random Sel." randomises verifier *scores* (the network is still trained), and
`2606.09646`'s "Random Labels Control" is a noise floor (our D4), which answers a different
question.

---

## 3. Method — the protocol

Each check names the **alternative explanation it rules out**, an operational definition, and
a threshold with a stated origin.

| | Check | Rules out | Criterion | Threshold from |
|---|---|---|---|---|
| **D1** | random-init lower bound | "any network of this architecture would do" | training share ≥ 50% of above-chance signal | Zhang & Bowman 2018; Voita & Titov 2020 |
| **D2** | attribute-irrelevant control | "the probe reads any salient edit" | AUC(control) < AUC(target) | Hewitt & Liang 2019 |
| **D3** | ceiling check | "no headroom left — trained and random are both at the top" | **real − random ≥ 2 × floor** | ours |
| **D4** | noise floor | "the effect is inside measurement noise" | effect > 2 × floor; floor definition declared **before** the run | ours |
| **D5** | cross-readout consistency | "the conclusion is an artefact of one readout" | two readouts agree | Elazar et al. 2020 (idea) |
| **D6** | bounded statistic in a saturated regime | "the statistic cannot see the effect" | if baseline is at a bound, the primary readout must be unbounded, declared in advance | ours |

**Provenance is deliberately explicit: D1–D2 are established checks that this line of work
has not applied; D3–D6 are ours.** §5 is what justifies the second group.

All thresholds and analysis choices were written down before the corresponding runs
(`PREREG.md`). Grouping is by scene, never by clip, so no scene layout appears in both
train and test folds.

---

## 4. Experiments

Wan2.2-TI2V-5B, 30-block DiT; features are block outputs, spatially pooled. Reference cell
layer 16 / t = 600, held fixed across datasets so layer choice cannot confound comparisons.
LikePhys: 750 physics clips, 12 scenarios × 10 subgroups. IntPhys 2 Main: 1012 clips,
253 scenes × 4.

### 4.1 Training contributes a minority of the signal (D1)

LikePhys, each source at its own best cell:

| Source | AUC |
|---|---|
| trained DiT | 0.9740 ± 0.0070 |
| **randomly initialised DiT** | **0.8576 ± 0.0268** |
| VAE latent | 0.8200 |
| raw pixel | 0.7886 |

Training share = (0.9740 − 0.8576)/(0.9740 − 0.5) = **24.6%** → **D1 fails**.

Two robustness checks:

- **Which initialisation?** Three independent random initialisations give a spread of
  **0.0051–0.0176** AUC; the trained-minus-random margin is 0.1164, about **7×** the largest
  spread. (The ±0.0268 above is across CV folds, a different quantity.)
- **Voita & Titov's own statistic.** Online prequential codelength: trained compression
  2.162, random 1.579 → training share **31.8%**. Both statistics agree that training
  contributes a minority, so this is not an artefact of using accuracy.
- **A non-linear readout.** A one-hidden-layer MLP gives 43.9%, still below 50%. This is a
  **weaker** control, not a stronger one: its absolute AUC (0.8776) is below the linear
  probe's (0.9686), of which 0.0213 is the PCA-256 step and 0.0697 is MLP underfitting at
  n = 750. We therefore claim only that a non-linear readout did not *narrow* the gap.

### 4.2 The probe is at least as sensitive to physics-irrelevant edits (D2)

The Introduction's example, plus the same comparison on LikePhys at five timesteps
(appearance ≥ physics at all five; at layer 16 / t = 600, 1.0000 vs 0.9686).

> Reading LikePhys's paper changed how we treat our own result: its colour change happens
> *mid-flight*, so it carries a temporal discontinuity, and the paper lists it as an
> **invalid variant**, not a control. The IntPhys 2 control we built ourselves has no
> temporal component and is the cleaner evidence.

**What this supports directly:** "high probe AUC ⇒ the representation encodes physics" does
not hold. **What it does not establish on its own:** that the probe reads edit artefacts —
that is an interpretation, supported in §4.3 and §4.4.

### 4.3 Lower bounds differ across benchmarks, consistently with how violations are made

Reference cell, both datasets:

| Lower bound | LikePhys | IntPhys 2 |
|---|---|---|
| raw pixel | **0.7886** | 0.5476 |
| VAE latent | 0.8200 | 0.5358 |
| randomly initialised DiT | 0.8353 | 0.5168 |
| trained DiT | 0.9686 | 0.6458 |

On LikePhys **raw pixels alone** reach 0.7886; on IntPhys 2 pixels and VAE latents sit near
chance. The gap in low-level separability is about **0.24 AUC**. This is measured.

The most economical explanation is **consistent with** a known construction difference —
LikePhys violations are edits applied to legal videos, IntPhys 2 violations are rendered
events — but we have **not** run the intervention that would establish it (generating the
same physical event both ways). Scene content, engine, resolution and duration also differ
and are uncontrolled. **We state this as an explanation, not a demonstrated cause.**

> This also resolves one tension: Invisible Hand reports the signal is "absent from the VAE
> latent input", which our LikePhys value of 0.8200 appeared to contradict. Moving to
> IntPhys 2 — the successor of the benchmark they used — gives 0.5358, consistent with their
> claim. The discrepancy is specific to LikePhys.

### 4.4 The other readout is largely a complexity meter (D5)

On LikePhys, denoising error is *lower* for violations (630 pairs, mean −0.01137,
p = 6.8e−14) — the opposite of the expected direction. A single complexity proxy
(mean temporal gradient) explains **56.3%** of the paired variance (r = +0.750). Matching
complexity **within scenario** reverses the sign: **+0.00112**. 84% of the negative effect
comes from two fluid scenarios, and 98% of it disappears or flips under matching.

So the two readouts on the same representation disagree about what counts as a confound, and
we can say what each is tracking: the probe responds to edits, the denoising error to scene
complexity. **Neither is a physics measurement.**

> A methodological by-product: the pooled regression intercept is −0.00900 (p ≈ 0, apparently
> "surviving"), while within-scenario matching gives +0.00112. Pooling forces one slope across
> twelve scenarios whose within-scenario R² ranges 0.41–0.98, so its intercept absorbs
> between-scenario heterogeneity. We report per-scenario throughout.

---

## 5. Protocol Validity

Rather than argue in prose that the six checks are right, we made the protocol itself the
object of experiment — the way Hewitt and Liang established control tasks.

### 5.1 Minimality — is any check redundant?

Remove one check at a time and count units whose verdict flips from "untrustworthy" to
"trustworthy", over the 12 LikePhys + 5 IntPhys 2 = **17 analysis units**:

| Removed | D1 | D2 | D3 | D4 | D5 | D6 |
|---|---|---|---|---|---|---|
| units missed | **10** | 6 | 8 | 3 | 1 | 7 |

No check is removable. D5 catches only one unit, but that unit is the headline denoising
effect of §4.4.

**D3 and D6 are not the same check.** The seven scenarios in our camera-motion experiment are
exactly the saturated ones D3 would discard — yet the question being asked requires measuring
*on* them. There, AUC reports a maximum drop of 0.0200 ("no decay") while d′ declines in
**7/7** and retains 0.754. Dropping units is not a substitute for changing the statistic.

### 5.2 Construct validity — does it get known answers right?

Four synthetic representations whose ground truth is fixed by construction, evaluated under
the real grouping and folds:

| Case | Ground truth | Protocol should | Result |
|---|---|---|---|
| A1 positive | encodes physics | all six pass | ✅ |
| A2 negative | appearance only | D1 or D2 must fail | ✅ |
| A3 mixed | physics + edit artefact | D2 fails, D1 passes | ✅ |
| A4 saturated | inflated numbers | D3 or D6 must fire | ✅ |

**4/4.** A2 also exposed a real defect in our own D1: with real AUC at 0.5326 the ratio's
denominator nearly vanishes and D1 reported a training share of **1.113** — "111% of an
effect that does not exist" — and passed. We added an interpretability guard (the ratio is
only read when the effect itself clears the floor). The same defect is live in real data:
IntPhys 2's `solidity` unit has AUC 0.5641 and reported 60.7%; it is now excluded for failing
D4.

### 5.3 ⚠️ Sufficiency — we found a counterexample

We then attacked our own protocol: construct a representation that passes all six checks but,
by construction, encodes no physics.

**C2 succeeded.** A "pipeline leak" — the two classes carrying a trace of different render
passes — reaches AUC 0.8100 on the trained side, 0.5317 on the random side, and 0.5085 on the
appearance control. It passes D1 (training-dependent), D2 (independent of appearance), D3/D6
(not saturated), D4 (above floor) and D5 (both readouts agree).

**Consequence: we withdraw the word "sufficient".** The protocol is **minimal necessary** —
each check is required, and passing all six is *not* enough. The failure mode we can now name
explicitly: violations and legal samples differing systematically in acquisition or rendering,
where that difference is orthogonal to both appearance and complexity.

What C2 establishes is a **logical** limit (a counterexample exists). Whether such leakage is
common in real benchmarks we have **not** verified; the 0.24 AUC pixel-level gap in §4.3 is a
hint, not a demonstration. A dedicated check for it (D7, pipeline control) is future work and
is deliberately **not** added to the protocol — adding an unvalidated check would repeat the
mistake we are criticising.

> A second attack, C1, did not get through. We report it as an **invalid experiment rather
> than a win**: the construction made the pseudo-signal nearly independent of the label, so
> it never tested what it was meant to.

---

## 6. Conclusion and Limitations

**Claim.** On the benchmarks and model we examined, the standard evidence for
"representations encode physical plausibility" does not survive the checks a probing result
should pass. We contribute D1–D6, evidence that each is necessary, calibration against known
answers, and a counterexample showing the set is not sufficient.

**What we are not claiming.** Not that video diffusion models lack physical knowledge. Not
that these benchmarks are worthless — IntPhys 2's near-chance lower bounds are exactly what a
well-constructed benchmark should look like, and that contrast is half our evidence.

### Limitations

1. **One backbone.** All representations come from Wan2.2-TI2V-5B. The *protocol* is stable
   across three orthogonal directions — two datasets giving opposite D1 verdicts (24.6% vs
   88.5%), two readouts disagreeing, two statistics agreeing — but whether the specific
   numbers generalise to other video diffusion models is **not verified**.
2. **C2's real-world prevalence is unverified** (§5.3).
3. **§4.3 is correlational.** The editing-vs-rendering account is an explanation consistent
   with the data, not an established cause; no intervention was run.
4. **The non-linear control is weak** (§4.1) — a stronger readout could in principle change
   D1's verdict.
5. **IntPhys 2's D1 passes at 88.5%, but with low absolute AUC** (0.5641–0.6458). That means
   "most of the little signal that exists is training-dependent", not "the probe works well".

> **On process.** Eleven corrections to our own analysis are recorded in the report, two of
> which changed the protocol: a readout chosen because its baseline had sd = 0.0000 turned out
> to be saturated rather than sensitive (this produced D6), and construct validity exposed the
> D1 degeneracy above. A project arguing that measurement is easy to get wrong should not hide
> its own instances.

---

## 7. Code and Reproducibility

Public repository layout (no cached features, no third-party videos, no internal paths):

```
src/ scripts/        extraction, probing, protocol checks
repro/l0_smoke.py    L0 smoke reproduction
results/*.json       all reported numbers, with field paths
results/figures/     F1–F8
PREREG.md            criteria, frozen before the runs
```

**L0 smoke path** — one GPU, ≤ 24 GB, under 30 minutes, from download to one comparable
number. It reproduces the ordering of the Introduction's example on a 24-scene subset:
appearance ≫ physics, and trained only modestly above random.

L0 checks the **ordering**, not the decimals. Measured over five independent scene draws:

| quantity | 24-scene subset | full set | reproducible? |
|---|---|---|---|
| appearance AUC | **1.0000, sd 0.0000** (5/5) | 1.0000 | exactly |
| physics AUC (trained) | 0.5598 ± 0.0339 | 0.6458 | as an interval |
| **D1 training share** | **0.788 ± 0.866, range [−0.636, +1.552]** | 0.885 | **no** |

We report this rather than a flattering seed: at 24 scenes the D1 ratio is unusable — on one
draw the random-init probe scored *above* the trained one, making the share negative. That is
the same denominator degeneracy our own construct-validity case A2 exposed (§5.2), reappearing
in real data. **L0's pass criterion is therefore three qualitative checks, not a number**:
appearance ≫ physics, physics above chance, random below trained. All three held on 5/5 draws.

Incidentally this is itself evidence: the physics signal is weak enough to need the full
dataset, while the appearance signal saturates on 96 clips.

> **Not yet verified on a clean machine.** Our own runs cannot distinguish "works anywhere"
> from "works because this machine already has the data, the dependencies and the environment
> variables". The assumptions are listed explicitly and need an independent run to confirm.

---

### Status

Experiments run: E0 (noise floor), E1 (lower bounds), E3 (readout grid), E4 (complexity
matching), E6 (literature audit), E7 (IntPhys 2 transfer), E10 (camera motion), T10 (protocol
validity). Numbers audited field-by-field against the result files (64 assertions).
Remaining: second backbone (E8), recomputation excluding saturated scenarios (E9).
