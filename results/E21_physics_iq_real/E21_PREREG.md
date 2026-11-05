# E21 PREREG — real-world video: is a "physics" probe distinguishing physics, or just camera?

**Written 2026-11-05, before any Physics-IQ feature was extracted.**

## 0. The gap this closes

`NEXT_STEPS_AND_DIVISION.md` names **"换 benchmark（真实世界视频）"** as the project's only
substantive remaining gap, and `E19_REPORT.md` §2.3(a) repeats it: *every* dataset in this
project — IntPhys 2 and LikePhys — is **synthetically edited**. Both construct an "impossible"
clip by modifying a rendered or synthetic source. That means every D1/D2 conclusion so far is
open to one structural objection:

> Perhaps these probes do not read physics at all; perhaps they read *the edit*. A colour
> shift, a teleported object, a duplicated mesh — all are artefacts of a rendering pipeline,
> and a linear probe on a large feature vector will find any consistent artefact.

D2 was designed to test exactly that, and it fails 34/34. But D2's control is a recolour we
applied ourselves. **No experiment in this project has ever used a video that was never
edited.** Physics-IQ contains 396 such clips: real cameras, real objects, no rendering.

## 1. What Physics-IQ can and cannot support here — established before designing

Checked against the official repo (`google-deepmind/physics-IQ-benchmark`) and the 396 files
on disk:

- Each of **66 scenes** is filmed from **3 perspectives**, and **each scene is recorded twice**
  (`take-1`, `take-2`) to capture *natural physical variation*. Official wording:
  *"Each scenario is recorded twice to capture natural physical variations, with some intended
  exceptions."*
- Therefore **`take-1` vs `take-2` is NOT a possible/impossible pair.** Both takes are
  physically legal. The project's earlier note that Physics-IQ "needs possible/impossible
  paired labels" is correct for any *violation-detection* task, and this task does not exist in
  the data.
- Every scene–perspective key is complete: **198/198 keys have both takes**, 66 scenes ×
  3 perspectives, 30 FPS, 240 frames (8 s) each.

**Consequence: this experiment cannot answer "does the probe detect physical
impossibility in real video".** It is not a replacement for D1 on a third dataset and will
not be written as one. Anyone reading E21 as "physics understanding on real video" has
misread it.

## 2. What it CAN answer, and why that is worth a GPU run

The take axis is a **physics-preserving** manipulation of the real world — the strongest
version of D2's control that exists, because *we did not create it*:

| task | what varies | physics | who made it |
|---|---|---|---|
| D1 (IntPhys 2 / LikePhys) | possible vs impossible | **changes** | benchmark author |
| D2 control (recolour) | hue/saturation only | legal | **us** |
| **E21 take task** | **two different physical performances of the same scene** | **legal; both real** | **the world** |

If the same frozen probe separates `take-1` from `take-2` at an AUC near its physics AUC,
then "the probe reads a consistent, physics-irrelevant property of the clip" stops being a
hypothesis about synthetic artefacts and becomes a property of **real sensor data** as well.
That is a genuinely new row in the project's evidence table, and it is obtainable from 396
videos on one GPU.

## 3. Design (frozen)

**Data.** All 396 clips: 66 scenes × 3 perspectives × 2 takes, 30 FPS, 8 s. Clips are read
the same way E7 reads IntPhys 2 — **49 frames spanning the whole clip** (`linspace`), because
the event is not annotated. Size 512, CogVideoX-compatible preprocessing.

**Backbone.** Wan2.1-T2V-1.3B, matching the project's reference backbone and its 30 layers.
L16 / t=600, feature seeds 0/1/2 — the same reference cell as everything else, so the numbers
are comparable to E8's IntPhys 2 row.

**Probe.** `e8_analyse.probe` imported verbatim, never re-implemented (E11: a rewrite shifts
AUC by 2.3 pp). Grouped 5-fold CV.

**Grouping.** `scene` (66 groups). Both takes of a scene, and all three perspectives, stay in
one fold. This is non-negotiable: if a scene's take-1 were in train and its take-2 in test, the
probe could memorise the scene and the take task would be trivial for the wrong reason.

**Three arms, same features:**

| arm | labels | reads |
|---|---|---|
| `take` | take-1 = 0, take-2 = 1 | physics-irrelevant real variation |
| `perspective` | left/center/right split into 3 classes → binarised as left vs right only | camera pose (positive control: must be learnable) |
| `scene` (optional) | scene identity, 66 classes | sanity ceiling |

**Pre-registered comparison, one number.** `R = AUC_take / AUC_physics_on_IntPhys2` where
`AUC_physics_on_IntPhys2` is the **already-delivered** Wan2.1-1.3B value from
`E8_cells_intphys2.json` (0.6502, quoted read-only; E21 does not recompute it).

## 4. Verdict rules (written before running)

| id | rule | meaning if it holds |
|---|---|---|
| **D-take-1** | `AUC_take ≥ 0.60` | the probe separates two legal recordings of the same scene — real, physics-irrelevant variation is linearly readable |
| **D-take-2** | `AUC_take ≥ AUC_physics − 0.05` | the take signal is *at least as strong* as the physics signal. This is the strong claim and the one worth reporting |
| **D-take-3** | `AUC_perspective ≥ 0.95` | positive control: the pipeline can learn an obvious real-world difference, so a null on `take` is a finding, not a broken harness |

Decision boundaries are fixed here. `0.05` is E8 §15.2's weak-signal margin, reused rather
than invented.

**A null result is a real result and will be reported as one.** If `AUC_take ≈ 0.5`, then the
"probe reads artefacts" worry loses much of its force on real data, and that is a *stronger*
paper than another failure.

## 5. Mandatory controls before any number is quoted

1. **Preflight** on the GPU (`W.preflight`), as every other task does — a defective SM
   produces silently wrong numbers.
2. **Raw-pixel and VAE-latent lower bounds** on the same take task. Without them a high
   `AUC_take` could be a property of pixels, not of the trained transformer.
3. **Random-init backbone** on the same take task. This is what makes the result interpretable
   in the project's own terms: if random-init separates takes as well as trained, the take
   signal is not learned physics either way.
4. **Identity self-check** of the reader: the same clip read twice must give a bit-identical
   feature vector.

## 6. Deliverables

```
results/E21_physics_iq_real/
    E21_PREREG.md          this file
    manifest.json          396 clips with take / scene / perspective, sha256 of each file
    E21_cells.json         take / perspective / lower-bound / random-init cells
    E21_verdict.json       D-take-1..3 with the threshold each was judged against
    E21_REPORT.md          results, the honest scope statement, and the limitation
    features/              per-clip .pt records (git-ignored, like every other cache)
```

## 7. Stated limitations, to appear in the report unprompted

- **No violation labels exist** in this data. E21 measures a *contrast*, not detection. It
  cannot be compared to Physics-IQ leaderboard scores (which come from a generation-and-scoring
  pipeline, not from probing real footage).
- 66 scenes is small; 3 perspectives per scene are **not** independent, so the effective n for
  any scene-level claim is 66, not 198.
- `take-1`/`take-2` differ in more than physics: lighting drift, camera micro-motion, object
  placement. A high `AUC_take` therefore does **not** identify *which* nuisance the probe
  reads. It only shows that a physics-irrelevant nuisance is readable — which is the claim.
