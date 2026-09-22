"""Geometry self-checks for the S4 warp, run before any extraction.

Each check targets a specific way the warp could be silently wrong -- the class of bug
that PITFALLS #1-#5 is about: no exception, just wrong numbers downstream.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s4_camera import (CROP_PX, CROP_SCALE, FRAMES, LEVELS, SIZE, apply_camera,
                       build_grid, crop_geometry, trajectory_seed)

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


print("=== 1. excursion matches the frozen A.5 levels ===")
for lvl, cfg in LEVELS.items():
    _, r = build_grid(lvl, "fluid", "subgroup_000", DEV)
    want = cfg["cum_px"]
    got = r["translation_px_excursion"]
    # jitter adds a little; allow 4*jitter slack
    tol = max(0.5, 4 * cfg["jitter_px"])
    check(f"level {lvl} excursion {got:7.2f} px (target {want})", abs(got - want) <= tol)
    check(f"level {lvl} rotation  {r['rot_deg_excursion']:6.3f} deg "
          f"(target {cfg['rot_deg_total']})",
          abs(r["rot_deg_excursion"] - cfg["rot_deg_total"]) < 1e-3)

print("\n=== 2. no sample ever leaves the frame (=> zero padding) ===")
worst = 0.0
for lvl in LEVELS:
    for sg in range(10):
        g, r = build_grid(lvl, "fluid", f"subgroup_{sg:03d}", DEV)
        worst = max(worst, r["max_abs_grid_coord"])
check(f"max |grid coord| over all levels x 10 subgroups = {worst:.4f} <= 1.0", worst <= 1.0)
check(f"margin left = {(1 - worst) * 256:.1f} px", worst < 1.0)

print("\n=== 3. same trajectory for valid and violation of one subgroup ===")
g1, _ = build_grid(3, "fluid", "subgroup_004", DEV)
g2, _ = build_grid(3, "fluid", "subgroup_004", DEV)
check("repeat call is bit-identical", torch.equal(g1, g2))
g3, _ = build_grid(3, "fluid", "subgroup_005", DEV)
check("different subgroup gets a different path", not torch.equal(g1, g3))
g4, _ = build_grid(3, "river", "subgroup_004", DEV)
check("different scenario gets a different path", not torch.equal(g1, g4))

print("\n=== 4. level 0 is an identity apart from the shared crop ===")
g0, r0 = build_grid(0, "fluid", "subgroup_000", DEV)
check(f"level 0 excursion == 0", r0["translation_px_excursion"] == 0.0)
check(f"level 0 rotation == 0", r0["rot_deg_excursion"] == 0.0)
# grid must be the same for every frame at level 0
check("level 0 grid identical across frames", torch.equal(g0[0], g0[-1]))
# and it must be exactly the centre crop
lin = torch.linspace(-1.0, 1.0, SIZE)
vv, uu = torch.meshgrid(lin, lin, indexing="ij")
expect = torch.stack([CROP_SCALE * uu, CROP_SCALE * vv], dim=-1).to(DEV)
check("level 0 grid == pure centre crop", torch.allclose(g0[0], expect, atol=1e-6))

print("\n=== 5. warp actually moves content, and by the right amount ===")
# a sharp vertical edge: after a pure horizontal pan the edge must shift
px = torch.full((1, 3, FRAMES, SIZE, SIZE), -1.0, device=DEV)
px[..., SIZE // 2:] = 1.0


def edge_col(img):
    prof = img.mean(dim=0)[SIZE // 2]
    return int((prof > 0).nonzero()[0].item())


for lvl in (0, 3):
    g, r = build_grid(lvl, "fluid", "subgroup_000", DEV)
    w = apply_camera(px, g)
    e0 = edge_col(w[0, :, 0])
    e48 = edge_col(w[0, :, -1])
    shift_out = abs(e48 - e0)
    # output pixels are CROP_SCALE source-px each, so a source shift of D px shows up as
    # D / CROP_SCALE output px; the horizontal component is excursion * |cos(dir)|
    import math
    expect_px = r["translation_px_excursion"] * abs(math.cos(math.radians(
        r["pan_direction_deg"]))) / CROP_SCALE
    print(f"    level {lvl}: edge moved {shift_out} out-px, expected ~{expect_px:.1f}")
    if lvl == 0:
        check("level 0 leaves the edge where it is", shift_out == 0)
    else:
        check("level 3 moves the edge by the predicted amount",
              abs(shift_out - expect_px) <= max(3.0, 0.1 * expect_px))

print("\n=== 6. no black border introduced (padding would show as -1 ramp) ===")
# use a constant mid-grey image: any padded pixel would read 0.0 instead of 0.5
flat = torch.full((1, 3, FRAMES, SIZE, SIZE), 0.5, device=DEV)
for lvl in LEVELS:
    g, _ = build_grid(lvl, "fluid", "subgroup_000", DEV)
    w = apply_camera(flat, g)
    check(f"level {lvl}: warped min == 0.5 (no padded zeros)",
          torch.allclose(w.min(), torch.tensor(0.5, device=DEV), atol=1e-5),
          f"min={float(w.min()):.6f}")

print("\n=== 7. crop geometry reported for A.3 item 3 ===")
cg = crop_geometry(3)
print(f"    crop {cg['crop_px']}px of {SIZE}px -> linear {cg['crop_scale_linear']:.4f}, "
      f"FOV area {cg['effective_fov_area_frac']*100:.1f}%, "
      f"upsample {cg['upsample_factor']:.4f}x")
check("crop is shared by all levels", cg["crop_shared_across_levels"])

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
