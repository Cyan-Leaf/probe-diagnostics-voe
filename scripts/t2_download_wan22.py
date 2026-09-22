"""T2: download Wan2.2-TI2V-5B-Diffusers into the persistent HF cache (resumable).

Run (background):
  export http_proxy=... https_proxy=... HF_HOME=$HF_HOME
  nohup uv run --no-project python scripts/t2_download_wan22.py > results/T2_env/download_wan22.log 2>&1 &
"""

import os
import time
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
REVISION = "b8fff7315c768468a5333511427288870b2e9635"  # pinned: sha fetched 2026-09-16


def main() -> None:
    assert os.environ.get("https_proxy"), "export the oversea proxy first"
    hf_home = os.environ.get("HF_HOME")
    assert hf_home, "set HF_HOME to the persistent cache dir"
    Path(hf_home).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    path = snapshot_download(
        repo_id=REPO,
        revision=REVISION,
        allow_patterns=[
            "model_index.json",
            "README.md",
            "scheduler/*",
            "tokenizer/*",
            "text_encoder/*",
            "transformer/*",
            "vae/*",
        ],
        max_workers=8,
    )
    print(f"done in {time.time()-t0:.0f}s -> {path}", flush=True)


if __name__ == "__main__":
    main()
