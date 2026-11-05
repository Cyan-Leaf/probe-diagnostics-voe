# What Do Physics Probes Actually Measure?

### A Diagnostic Protocol and Its Limits

Code and results for a course project on **what a probe AUC does and does not license you to
conclude** about a video diffusion model's internal representations.

---

## The one-paragraph version

Take one model, one set of clips, one linear classifier, one layer, one timestep. Change
nothing but what the labels mean:

| Task on IntPhys 2 (identical features and folds) | Probe AUC |
|---|---|
| physically possible vs impossible | **0.6458** |
| possible vs *the same clips recoloured* (physics untouched) | **1.0000** |

The recolouring is physically legal — identical HSV shift on every frame, V channel
unchanged to 0.000000, frame-difference profile correlating 0.9989 with the original.

This does **not** show the model fails at physics. It shows that **0.6458 cannot be used to
support "this representation encodes physical plausibility"**, because the same probe scores
*higher* on a manipulation containing no physics at all.

---

## Where the project stands now (2026-11-05)

The two checks above behave very differently under replication, and that asymmetry is the
result:

| | Result | Scale |
|---|---|---|
| **D2** (the recolour control) | **fails in 34/34 cells, zero exceptions** | 7 model families, 14 variants, 2 datasets |
| **D1** (the random-init bound) | **dataset-dependent**: 24 pass on IntPhys 2, 8 fail on LikePhys | same cells |

D2 failing everywhere makes "the probe reads the edit, not the physics" **systematic** rather
than a quirk of one model: it survives a non-generative encoder (V-JEPA 2), a 33 B model with
a different tokenizer and VAE (MiniMax-H3), a MoE pair of experts, five token budgets, a
non-linear readout, and both datasets. D1 changing sign between datasets means it **cannot** be
stated as a universal claim about training — it is a property of the benchmark's construction.

Two later results sharpen the scope:

- **The probe is blind to a natural, physics-irrelevant nuisance in real footage.** On 396 real
  Physics-IQ clips, separating the two takes of the *same* scene scores AUC **0.4891**
  (direction-free separation 0.5201, i.e. chance), while a positive control — camera direction —
  scores **0.9717** on the same features. So D2's
  failure is not "linear probes read anything"; it is "linear probes read the statistical
  residue of synthetic editing". (`results/E21_physics_iq_real/`)

  This is **not** a physics-violation benchmark on real video, and the report says so: Physics-IQ
  records every scene twice, so `take-1`/`take-2` are two *legal* performances, not a
  possible/impossible pair. That task does not exist in this data.

- **Every verdict now carries a cluster-bootstrap interval**, across all 69 cells. D1 is a
  *ratio*, and in the weak-signal region its denominator was observed as low as 0.096, which can
  turn an AUC gap of 0.0116 into ~10 pp of training share. Result: **D2 is robust in 37/37**
  (smallest lower bound **+0.0225**, still clear of 0); **D1 is robust in 31/37**. All six
  fragile cells sit on the 50% boundary — CogVideoX-2b/5b on LikePhys reach 54.06% and 51.75%;
  MiniMax-H3's L16 cell, reported as a pass at 63.87%, has the interval [42.30, 76.34] and so
  cannot be separated from a coin flip on this evidence. That cell had already been flagged as
  weak-signal in E19; the interval confirms it from an independent axis.
  (`results/E20_bootstrap_ci/`, `E20ext_REPORT.md`)

---

## What is here

```
scripts/           feature extraction, probing, each protocol check
repro/l0_smoke.py  L0 smoke reproduction (one GPU, <30 min)
results/           every reported number as JSON, with field paths
results/T13_figures/   F1-F8
PREREG.md          criteria, frozen before the corresponding runs
SPEC.md            full specification
EXPERIMENTS.md     the experiments chapter
PROTOCOL_VALIDITY.md   how the protocol itself was tested
```

Not here, by design: cached features (21 GB of `.pt`/`.npy`), third-party videos, fetched
paper PDFs. Datasets are downloaded by script; see below.

---

## The protocol

Six checks a representation-level physics claim should pass. Each names the alternative
explanation it rules out.

| | Check | Rules out |
|---|---|---|
| D1 | random-init lower bound | "any network of this architecture would do" |
| D2 | attribute-irrelevant control | "the probe reads any salient edit" |
| D3 | ceiling check | "this unit has no resolution left" |
| D4 | noise floor | "the effect is inside measurement noise" |
| D5 | cross-readout consistency | "it is an artefact of one readout" |
| D6 | bounded statistic in a saturated regime | "the statistic cannot see the effect" |

**D1–D2 are established checks** (Hewitt & Liang 2019; Zhang & Bowman 2018 via Voita &
Titov 2020) that this line of work has not applied — we audited four papers and found zero
hits for `untrained` / `from scratch` in all of them. **D3–D6 are ours**, and
`PROTOCOL_VALIDITY.md` is the evidence for them.

### The protocol is `minimal necessary`, not sufficient

Each check catches something the other five miss (17-unit ablation), and all four
known-answer construct-validity cases come out right. But we also built a representation that
**passes all six and encodes no physics** — a "pipeline leak" where the two classes carry
traces of different render passes (AUC 0.8100 trained / 0.5317 random / 0.5085 appearance
control).

So: passing all six is necessary, not sufficient. The named failure mode is violations and
legal samples differing in acquisition or rendering, orthogonally to both appearance and
complexity.

---

## Reproducing

### L0 smoke path — one GPU, under 30 minutes

```bash
pip install -r requirements.txt

# IntPhys 2 (~12 GB)
huggingface-cli download facebook/IntPhys2 --repo-type dataset --local-dir data/IntPhys2

python repro/l0_smoke.py --dry-run     # print the plan, no compute
python repro/l0_smoke.py --scenes 24   # the real run
```

Reproduces the ordering of the example above on a 24-scene subset: appearance ≫ physics, and
trained only modestly above random.

**L0 checks the ordering, not the decimals.** Subset AUCs differ from full-set values; the
measured spread over five independent scene draws is in `T14_REPRO.md`, together with every
assumption L0 makes about its environment.

> **Not yet verified on a clean machine.** Our own runs cannot distinguish "works anywhere"
> from "works because this machine already has the data, the dependencies and the environment
> variables set". An independent run is needed.

### Full pipeline

Extraction is the expensive part (~21 GB of features, several GPU-hours). See
`scripts/t3_extract.py` and `scripts/e7_extract.py`. All analysis scripts read the cache and
re-derive every number in `results/`.

---

## Notes on how to read the results

- **Numbers come in two flavours and must not be mixed**: per-source *best cell*
  (e.g. 24.6% training share) and the *fixed reference cell* layer 16 / t=600
  (28.5%). Both are correct; the difference is which cell.
- **`t >= 800` denoising deltas are not reported** — they fall inside the noise floor (D4).
- **Per-scenario beats the global mean.** In one case the pooled regression intercept is
  −0.00900 while within-scenario matching gives +0.00112, an actual sign reversal.
- **We do not report a human baseline.** The single-rater observation in the repo is
  stimulus-discriminability, labelled exploratory; see `EXPERIMENTS.md` §5.9.0 for why.

---

## Corrections

Eleven corrections to our own analysis are recorded in `EXPERIMENTS.md` §5.11. Two changed
the protocol:

- A readout was chosen *because* its baseline had sd = 0.0000 — that turned out to mean
  saturated, not sensitive. This produced **D6**.
- Construct validity case A2 exposed a degeneracy in **D1**: with AUC near chance the ratio's
  denominator nearly vanishes, and it reported a "111% training share" on a representation
  with no signal at all. D1 now carries an interpretability guard.

A project arguing that measurement is easy to get wrong should not hide its own instances.

---

## Licence and data

Code under this repository is for coursework review. IntPhys 2 is CC-BY-NC-4.0 (Meta);
LikePhys and model weights follow their own licences and are **not redistributed here**.
