"""T2: inspect the annotation format / layout of the downloaded datasets.

Writes a machine-readable summary to results/T2_env/data_inventory.json and prints a digest.

Run:
  export HF_HOME=$HF_HOME
  uv run --no-project python scripts/t2_inspect_data.py
"""

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

HUB = Path(os.environ["HF_HOME"]) / "hub"
OUT = Path(__file__).resolve().parents[1] / "results" / "T2_env"


def snap(repo_dirname: str) -> Path | None:
    base = HUB / repo_dirname / "snapshots"
    if not base.exists():
        return None
    subs = sorted(base.iterdir())
    return subs[0] if subs else None


def head(obj, n=2):
    if isinstance(obj, list):
        return [head(o, n) for o in obj[:n]]
    if isinstance(obj, dict):
        return {k: (head(v, n) if isinstance(v, (dict, list)) else v) for k, v in obj.items()}
    return obj


def inspect_likephys(root: Path) -> dict:
    vids = sorted(root.rglob("*.mp4"))
    scen = Counter(p.relative_to(root).parts[1] for p in vids)
    kinds = Counter(p.stem.rsplit("_", 1)[0] for p in vids)
    subgroups = defaultdict(set)
    for p in vids:
        parts = p.relative_to(root).parts
        subgroups[parts[1]].add(parts[2])
    return {
        "root": str(root),
        "n_videos": len(vids),
        "scenarios": dict(scen),
        "n_scenarios": len(scen),
        "clip_kinds": dict(kinds),
        "subgroups_per_scenario": {k: len(v) for k, v in sorted(subgroups.items())},
        "example_paths": [str(p.relative_to(root)) for p in vids[:6]],
        "other_files": [str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p.suffix != ".mp4"][:20],
    }


def inspect_intphys2(root: Path) -> dict:
    files = [p for p in root.rglob("*") if p.is_file()]
    by_suffix = Counter(p.suffix for p in files)
    csvs = [p for p in files if p.suffix == ".csv"]
    csv_head = {}
    for c in csvs[:5]:
        lines = c.read_text(errors="ignore").splitlines()[:3]
        csv_head[str(c.relative_to(root))] = lines
    return {
        "root": str(root),
        "n_files": len(files),
        "by_suffix": dict(by_suffix),
        "top_level": sorted({p.relative_to(root).parts[0] for p in files}),
        "csv_files": [str(c.relative_to(root)) for c in csvs],
        "csv_head": csv_head,
        "n_videos": sum(1 for p in files if p.suffix == ".mp4"),
    }


def inspect_physion(root: Path) -> dict:
    out: dict = {"root": str(root), "files": {}}
    for f in sorted(root.iterdir()):
        if f.suffix != ".json":
            continue
        data = json.loads(f.read_text())
        info: dict = {"size_MB": round(f.stat().st_size / 1e6, 2), "type": type(data).__name__}
        if isinstance(data, list):
            info["n_records"] = len(data)
            if data:
                rec = data[0]
                info["record_keys"] = list(rec.keys()) if isinstance(rec, dict) else str(type(rec))
                info["first_record_preview"] = json.loads(json.dumps(head(rec, 2))[:2500])
        elif isinstance(data, dict):
            info["top_keys"] = list(data.keys())[:30]
            info["n_top_keys"] = len(data)
            k0 = next(iter(data))
            info["first_value_preview"] = json.loads(json.dumps(head(data[k0], 2))[:2500])
        out["files"][f.name] = info
    readme = root / "README.md"
    if readme.exists():
        out["readme_head"] = readme.read_text(errors="ignore")[:2500]
    return out


def main() -> None:
    inv = {}
    lp = snap("datasets--JianhaoDYDY--LikePhys-Benchmark")
    if lp:
        inv["likephys"] = inspect_likephys(lp)
    ip = snap("datasets--facebook--IntPhys2")
    if ip:
        inv["intphys2"] = inspect_intphys2(ip)
    pe = snap("datasets--PhysionLabs--Physion-Eval")
    if pe:
        inv["physion_eval"] = inspect_physion(pe)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "data_inventory.json").write_text(json.dumps(inv, indent=2, ensure_ascii=False))
    print(json.dumps(inv, indent=2, ensure_ascii=False)[:7000])
    print(f"\n-> {OUT/'data_inventory.json'}")


if __name__ == "__main__":
    main()
