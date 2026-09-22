"""S4 synthetic camera motion -- the warp pipeline (PREREG appendix A.2, frozen).

Implements A.2 literally:

  pixel domain, before VAE encode ....... applied to the [-1,1] float tensor
  per-frame 2D affine, frame-uniform .... translation + rotation + jitter, no perspective
  bilinear, align_corners=False ......... single grid_sample, see note below
  warp then centre-crop to 512x512 ...... NO zero/replicate padding, ever
  crop ratio fixed across all levels .... set by the strongest level (level 3)
  resolution stays 512x512 .............. crop is upsampled back
  frames stay 49
  jitter uses a fixed seed, and the SAME trajectory is used by the valid and the
  violation clips of one (scenario, subgroup)

Why one grid_sample and not warp-then-crop-then-resize: three separate resamples would
stack three interpolation kernels and blur far more than the camera motion itself.  The
crop scale, the rotation and the translation are all affine, so they compose into a single
sampling grid:

    source_norm = S * R(theta_f) @ (u, v) + T_f

with S = crop_size / 512 (sampling a smaller central region == crop, and the output being
512 px == the upsample), R the per-frame rotation and T_f the per-frame translation in
normalised units (1.0 == 256 px from centre).  That is exactly one bilinear resample.

Trajectory convention: the ramp is centred, ramp(f) = f/(F-1) - 0.5, so the displacement
*relative to frame 0* spans the full target excursion (A.5 defines the levels by
"累计位移（最大偏移）" = excursion from frame 0) while |T| itself only reaches CUM/2.
Centring halves the crop the geometry demands at identical excursion.

Zero padding is not merely avoided by construction -- `build_grid` asserts that every
sampled coordinate lies inside [-1, 1], so a violation raises instead of silently
producing black borders.
"""

import hashlib
import math

import torch

FRAMES = 49
SIZE = 512

# A.5 frozen levels: excursion from frame 0, in pixels
LEVELS = {
    0: {"cum_px": 0.0, "rot_deg_total": 0.0, "jitter_px": 0.0},
    1: {"cum_px": 32.0, "rot_deg_total": 0.895, "jitter_px": 0.250},
    2: {"cum_px": 64.0, "rot_deg_total": 1.790, "jitter_px": 0.500},
    3: {"cum_px": 128.0, "rot_deg_total": 3.581, "jitter_px": 1.000},
}

# Crop is fixed by the strongest level and shared by every level (A.2).
# Bound: S*(|cos|+|sin|) + |T|_inf <= 1, with |T| = cum/2/256 + 3*jitter/256.
CROP_PX = 352
CROP_SCALE = CROP_PX / SIZE  # 0.6875


def trajectory_seed(scenario: str, subgroup: str) -> int:
    """Stable seed from (scenario, subgroup) only -- never from the clip.

    A.2: the valid and the violation clips of one subgroup must ride the *same* camera
    path, otherwise the paired difference picks up a difference in motion as well.
    """
    h = hashlib.sha256(f"{scenario}/{subgroup}".encode()).digest()
    return int.from_bytes(h[:8], "big")


def crop_geometry(level: int) -> dict:
    """A.3 item 3: the crop ratio and effective field of view actually applied."""
    return {
        "crop_px": CROP_PX,
        "crop_scale_linear": CROP_SCALE,
        "effective_fov_area_frac": CROP_SCALE ** 2,
        "upsample_factor": SIZE / CROP_PX,
        "crop_shared_across_levels": True,
        "set_by_level": 3,
    }


def build_grid(level: int, scenario: str, subgroup: str, device: str,
               frames: int = FRAMES, size: int = SIZE):
    """Sampling grid [F, size, size, 2] plus the motion actually realised."""
    cfg = LEVELS[level]
    g = torch.Generator(device="cpu").manual_seed(trajectory_seed(scenario, subgroup))

    # pan direction: shared within the subgroup, varied across subgroups so the result is
    # not an artefact of one particular direction
    ang = float(torch.rand(1, generator=g).item()) * 2.0 * math.pi
    dir_x, dir_y = math.cos(ang), math.sin(ang)

    f = torch.arange(frames, dtype=torch.float32)
    ramp = f / (frames - 1) - 0.5  # centred, spans 1.0

    tx_px = cfg["cum_px"] * ramp * dir_x
    ty_px = cfg["cum_px"] * ramp * dir_y
    rot = torch.deg2rad(torch.tensor(cfg["rot_deg_total"], dtype=torch.float32)) * ramp

    if cfg["jitter_px"] > 0:
        jit = torch.randn(frames, 2, generator=g) * cfg["jitter_px"]
        tx_px = tx_px + jit[:, 0]
        ty_px = ty_px + jit[:, 1]

    half = size / 2.0
    tx = tx_px / half
    ty = ty_px / half

    lin = torch.linspace(-1.0, 1.0, size)
    vv, uu = torch.meshgrid(lin, lin, indexing="ij")  # vv = y, uu = x

    grids = []
    for i in range(frames):
        c, s = torch.cos(rot[i]), torch.sin(rot[i])
        sx = CROP_SCALE * (c * uu - s * vv) + tx[i]
        sy = CROP_SCALE * (s * uu + c * vv) + ty[i]
        grids.append(torch.stack([sx, sy], dim=-1))
    grid = torch.stack(grids, dim=0)

    worst = float(grid.abs().max())
    assert worst <= 1.0, (
        f"sampling grid leaves the frame (max |coord| = {worst:.4f} > 1) for level={level} "
        f"{scenario}/{subgroup}: that would require padding, which A.2 forbids. "
        f"Reduce CROP_PX.")

    realised = {
        "pan_direction_deg": math.degrees(ang),
        "translation_px_excursion": float((tx_px.max() - tx_px.min()) ** 2
                                          + (ty_px.max() - ty_px.min()) ** 2) ** 0.5,
        "translation_px_per_frame_mean": float(
            torch.hypot(tx_px.diff(), ty_px.diff()).mean()) if frames > 1 else 0.0,
        "rot_deg_excursion": float(torch.rad2deg(rot.max() - rot.min())),
        "jitter_px_std": cfg["jitter_px"],
        "max_abs_grid_coord": worst,
        "padding_pixels": 0,
    }
    return grid.to(device), realised


@torch.no_grad()
def apply_camera(px: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """px [1,3,F,H,W] in [-1,1] -> warped [1,3,F,H,W], single bilinear resample.

    `grid_sample` is fed per-frame as a batch of [F,3,H,W]; padding_mode is set to
    "zeros" only as a formality -- `build_grid` has already asserted that no sample falls
    outside the frame, so no padded value is ever read.
    """
    x = px[0].permute(1, 0, 2, 3)  # [F,3,H,W]
    out = torch.nn.functional.grid_sample(
        x, grid.to(x.dtype), mode="bilinear", padding_mode="zeros", align_corners=False)
    return out.permute(1, 0, 2, 3).unsqueeze(0).contiguous()


def pooled_raw_pixel_from_px(px: torch.Tensor, size: int = 32) -> torch.Tensor:
    """raw_pixel_32 recomputed from the *warped* pixels.

    The unwarped version in t3_extract.py reads the original uint8 frames; for S4 the
    pixel baseline has to see the same camera motion as the DiT does, otherwise the
    raw_pixel lower bound would be measured on different data than the probe.
    """
    x = (px[0].permute(1, 0, 2, 3) + 1.0) * 0.5  # [F,3,H,W] in [0,1]
    x = torch.nn.functional.interpolate(x, size=(size, size), mode="bilinear",
                                        align_corners=False)
    groups = [x[0:1]] + [x[1 + 4 * i: 1 + 4 * (i + 1)] for i in range((x.shape[0] - 1) // 4)]
    per_latent = torch.stack([g.mean(dim=0) for g in groups], dim=0)
    return per_latent.flatten(1).half().cpu()
