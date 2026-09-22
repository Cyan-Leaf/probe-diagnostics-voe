"""Confirm the cuda:0 corruption with an independent memory test, and dump nvidia-smi health.

Run: uv run --no-project python scripts/t2_diag_gpu0.py
"""

import json
import subprocess
from pathlib import Path

import torch

OUT = Path(__file__).resolve().parents[1] / "results" / "T2_env"


def h2d_roundtrip(dev: str, mb: int, reps: int, seed: int = 0) -> dict:
    """Copy a known pattern to the GPU and back; count mismatching bytes."""
    g = torch.Generator().manual_seed(seed)
    n = mb * 1024 * 1024
    src = torch.randint(0, 256, (n,), dtype=torch.uint8, generator=g)
    bad = 0
    worst = 0
    for _ in range(reps):
        back = src.to(dev).cpu()
        diff = (back != src)
        bad += int(diff.sum())
        if int(diff.sum()):
            worst = max(worst, int((back.int() - src.int()).abs().max()))
    return {"mb": mb, "reps": reps, "mismatching_bytes": bad,
            "total_bytes": n * reps, "worst_abs_delta": worst}


def elementwise_selfcheck(dev: str, n: int = 1 << 24) -> dict:
    """y = x*2+1 computed on GPU vs closed form; x is exactly representable."""
    x = torch.arange(n, dtype=torch.float32) % 1000
    y_gpu = (x.to(dev) * 2.0 + 1.0).cpu()
    y_ref = x * 2.0 + 1.0
    d = (y_gpu - y_ref).abs()
    return {"n": n, "max_abs_diff": d.max().item(), "n_wrong": int((d > 0).sum())}


def main() -> None:
    rep: dict = {"torch": torch.__version__, "cuda_build": torch.version.cuda, "gpus": {}}
    for i in range(torch.cuda.device_count()):
        dev = f"cuda:{i}"
        print(f"=== {dev} ===", flush=True)
        r = {
            "name": torch.cuda.get_device_name(i),
            "h2d_64MB_x5": h2d_roundtrip(dev, 64, 5),
            "elementwise": elementwise_selfcheck(dev),
        }
        print(json.dumps(r, indent=2), flush=True)
        rep["gpus"][dev] = r

    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,pci.bus_id,ecc.errors.uncorrected.volatile.total,"
         "ecc.errors.corrected.volatile.total,retired_pages.single_bit_ecc.count,"
         "retired_pages.double_bit_ecc.count,temperature.gpu,power.draw,clocks_throttle_reasons.active",
         "--format=csv"],
        capture_output=True, text=True,
    )
    rep["nvidia_smi_health_csv"] = smi.stdout.strip().splitlines()
    print("\n".join(rep["nvidia_smi_health_csv"]))

    xid = subprocess.run(["bash", "-lc", "dmesg 2>/dev/null | grep -i -E 'xid|nvrm' | tail -20"],
                         capture_output=True, text=True)
    rep["dmesg_xid_tail"] = xid.stdout.strip().splitlines() or ["<no dmesg access or no entries>"]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gpu_health.json").write_text(json.dumps(rep, indent=2))
    print(f"\n-> {OUT/'gpu_health.json'}")

    bad = [d for d, r in rep["gpus"].items()
           if r["h2d_64MB_x5"]["mismatching_bytes"] or r["elementwise"]["n_wrong"]]
    print(f"\nFAULTY DEVICES: {bad or 'none'}")


if __name__ == "__main__":
    main()
