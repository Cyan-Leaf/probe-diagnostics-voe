"""Isolate the numerical corruption seen in preprocess_frames on GPU (T2 blocker).

Tests, per visible GPU: H2D copy of a uint8 tensor, elementwise scaling, permute+contiguous,
F.interpolate, and a matmul; each compared against the CPU result.

Run: uv run --no-project python scripts/t2_diag_gpu.py
"""

import json

import numpy as np
import torch


def check(name: str, gpu: torch.Tensor, cpu: torch.Tensor, out: dict) -> None:
    d = (gpu.detach().float().cpu() - cpu.detach().float()).abs().max().item()
    out[name] = {"max_abs_diff": d, "gpu_min": gpu.min().item(), "gpu_max": gpu.max().item(),
                 "cpu_min": cpu.min().item(), "cpu_max": cpu.max().item()}
    print(f"  {name:28s} max|gpu-cpu|={d:.6g}  gpu[{gpu.min():.4g},{gpu.max():.4g}] "
          f"cpu[{cpu.min():.4g},{cpu.max():.4g}]", flush=True)


def main() -> None:
    rng = np.random.default_rng(0)
    u8 = rng.integers(0, 256, size=(49, 512, 512, 3), dtype=np.uint8)
    t_cpu = torch.from_numpy(u8)
    report = {"torch": torch.__version__, "cuda_build": torch.version.cuda,
              "driver_gpus": torch.cuda.device_count(), "per_gpu": {}}

    for dev_i in range(torch.cuda.device_count()):
        dev = f"cuda:{dev_i}"
        print(f"\n=== {dev} ({torch.cuda.get_device_name(dev_i)}) ===", flush=True)
        out: dict = {}
        try:
            g = t_cpu.to(device=dev)
            check("h2d_uint8", g, t_cpu, out)

            g_f = g.float()
            check("uint8_to_float32", g_f, t_cpu.float(), out)

            g_s = g_f / 255.0
            check("div255", g_s, t_cpu.float() / 255.0, out)

            g_p = g_f.permute(0, 3, 1, 2)
            check("permute", g_p, t_cpu.float().permute(0, 3, 1, 2), out)

            g_pc = g_p.contiguous()
            check("permute_contiguous", g_pc, t_cpu.float().permute(0, 3, 1, 2).contiguous(), out)

            small = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).reshape(2, 3, 8, 8)
            gi = torch.nn.functional.interpolate(small.to(dev), size=(16, 16), mode="bilinear", align_corners=False)
            ci = torch.nn.functional.interpolate(small, size=(16, 16), mode="bilinear", align_corners=False)
            check("interpolate_small", gi, ci, out)

            big = t_cpu.float().permute(0, 3, 1, 2)[:4]
            gb = torch.nn.functional.interpolate(big.to(dev), size=(256, 256), mode="bilinear", align_corners=False)
            cb = torch.nn.functional.interpolate(big, size=(256, 256), mode="bilinear", align_corners=False)
            check("interpolate_big", gb, cb, out)

            a = torch.randn(1024, 1024)
            check("matmul_fp32", (a.to(dev) @ a.to(dev)), a @ a, out)

            report["per_gpu"][dev] = {"ok": True, **out}
        except Exception as exc:  # noqa: BLE001
            report["per_gpu"][dev] = {"ok": False, "err": f"{type(exc).__name__}: {exc}".splitlines()[0], **out}
            print(f"  FAILED: {exc}".splitlines()[0], flush=True)

    print("\n" + json.dumps(report, indent=2)[:4000])


if __name__ == "__main__":
    main()
