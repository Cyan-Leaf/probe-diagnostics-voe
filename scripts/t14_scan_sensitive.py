"""T14.2(a): scan the repository for anything that must not be published.

This runs before the proposal is written because its result can change what the proposal is
allowed to claim in §7. The categories are the ones HANDOFF names, plus a few that are easy
to leak by accident (tokens, IPs, the shared-mount prefix).

The scan reports; it does not edit. Anything found gets listed in T14_REPRO.md with a
decision, so that removing a path cannot silently change a documented command.

Run: python scripts/t14_scan_sensitive.py
"""

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "T14_release"

# Extensions worth scanning: text we would actually publish.
EXTS = {".md", ".py", ".sh", ".txt", ".json", ".yaml", ".yml", ".toml", ".cfg", ".html"}
SKIP_DIRS = {".git", ".venv", "__pycache__", "cache", "node_modules"}

PATTERNS = {
    "internal_hostname": (
        r"aiplatform-[\w.-]+|[\w-]+\.idchb\d\w*\.hb\d\.kwaidc\.com|[\w-]+\.kwaidc\.com",
        "开发机主机名"),
    "corp_domain": (r"corp\.kuaishou\.com|\.kuaishou\.com|gifshow\.com|txyun", "内网域名"),
    "shared_mount": (r"$WORKDIR[\w/.-]*", "共享盘绝对路径"),
    "home_path": (r"/home/[\w.-]+|/root/[\w./-]+", "家目录绝对路径"),
    "username": (r"<user>\w*", "用户名"),
    "gpu_uuid": (r"GPU-[0-9a-f]{8}-[0-9a-f]{4}", "GPU 设备 UUID"),
    "pci_bus": (r"0000:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.\d", "PCI 总线地址"),
    "proxy": (r"<http-proxy>\d*[\w.:-]*|:11080", "内网代理"),
    "ip_addr": (r"\b(?:10|172|192)\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "内网 IP"),
    "token_like": (r"(?i)(api[_-]?key|secret|token|password)\s*[=:]\s*['\"][^'\"]{8,}",
                   "疑似凭证"),
    "hf_token": (r"hf_[A-Za-z0-9]{20,}", "HuggingFace token"),
}


def files():
    for p in ROOT.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in EXTS:
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        yield p


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    compiled = {k: (re.compile(v[0]), v[1]) for k, v in PATTERNS.items()}
    hits = {k: [] for k in PATTERNS}
    n_files = 0

    for p in files():
        n_files += 1
        try:
            text = p.read_text(errors="ignore")
        except Exception:
            continue
        rel = str(p.relative_to(ROOT))
        for line_no, line in enumerate(text.splitlines(), 1):
            for key, (rx, _) in compiled.items():
                m = rx.search(line)
                if m:
                    hits[key].append({
                        "file": rel, "line": line_no,
                        "match": m.group(0)[:120],
                        "context": line.strip()[:160],
                    })

    summary = {k: len(v) for k, v in hits.items()}
    by_file = {}
    for k, v in hits.items():
        for h in v:
            by_file.setdefault(h["file"], []).append(k)

    payload = {
        "n_files_scanned": n_files,
        "counts": summary,
        "total_hits": sum(summary.values()),
        "files_with_hits": {f: sorted(set(k)) for f, k in sorted(by_file.items())},
        "hits": hits,
        "note": ("报告，不改动。每条需在 T14_REPRO.md 给出处理决定 —— "
                 "直接删路径会让文档里记录的命令失效，必须替换为占位符或环境变量。"),
    }
    (OUT / "T14_sensitive_scan.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False))

    print(f"scanned {n_files} text files\n")
    print(f"{'category':20s} {'hits':>6s}  what")
    for k, (_, label) in compiled.items():
        print(f"{k:20s} {summary[k]:6d}  {label}")
    print(f"\ntotal: {sum(summary.values())} hits in {len(by_file)} files")

    print("\n=== files with the most exposure ===")
    for f, ks in sorted(by_file.items(), key=lambda kv: -len(kv[1]))[:15]:
        print(f"  {len(ks):3d}  {f}   {sorted(set(ks))}")

    for k in ("token_like", "hf_token", "gpu_uuid", "ip_addr"):
        if summary[k]:
            print(f"\n!!! {k} ({summary[k]}) -- must be resolved before publishing")
            for h in hits[k][:5]:
                print(f"    {h['file']}:{h['line']}  {h['context'][:110]}")

    print(f"\nwrote {OUT/'T14_sensitive_scan.json'}")


if __name__ == "__main__":
    main()
