"""T2: precompute UMT5 prompt embeddings offline (MODELS.md risk #8).

The 11 GB text encoder must never be resident during probing runs, so we compute the
embeddings once (empty prompt + any scenario prompts) and store them as fp16 tensors.

Run:
  export HF_HOME=$HF_HOME
  uv run --no-project python scripts/t2_prompt_embeds.py --out results/T2_env/prompt_embeds.pt
"""

import argparse
import json
from pathlib import Path

import torch
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput  # noqa: F401  (import check)
from transformers import AutoTokenizer, UMT5EncoderModel

REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
REVISION = "b8fff7315c768468a5333511427288870b2e9635"
MAX_LEN = 226


def prompt_clean(text: str) -> str:
    """Mirrors diffusers.pipelines.wan.pipeline_wan.prompt_clean (whitespace/basic cleanup)."""
    import html
    import re

    import ftfy

    text = ftfy.fix_text(html.unescape(html.unescape(str(text))))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", nargs="*", default=[""])
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import wan_probe_lib as W

    pre = W.preflight(args.device)
    print("preflight:", json.dumps(pre), flush=True)

    tok = AutoTokenizer.from_pretrained(REPO, revision=REVISION, subfolder="tokenizer")
    enc = UMT5EncoderModel.from_pretrained(
        REPO, revision=REVISION, subfolder="text_encoder", torch_dtype=torch.bfloat16
    ).to(args.device).eval()

    prompts = [prompt_clean(p) for p in args.prompts]
    ti = tok(
        prompts,
        padding="max_length",
        max_length=MAX_LEN,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    ids, mask = ti.input_ids, ti.attention_mask
    seq_lens = mask.gt(0).sum(dim=1).long()
    out = enc(ids.to(args.device), mask.to(args.device)).last_hidden_state
    out = [u[:v] for u, v in zip(out, seq_lens)]
    embeds = torch.stack(
        [torch.cat([u, u.new_zeros(MAX_LEN - u.size(0), u.size(1))]) for u in out], dim=0
    )

    dst = Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "prompts": prompts,
            "embeds": embeds.to(torch.float16).cpu(),
            "seq_lens": seq_lens.cpu(),
            "repo": REPO,
            "revision": REVISION,
            "max_sequence_length": MAX_LEN,
        },
        dst,
    )
    meta = {
        "out": str(dst),
        "shape": list(embeds.shape),
        "dtype_saved": "float16",
        "prompts": prompts,
        "seq_lens": seq_lens.tolist(),
        "peak_vram_GB": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "preflight": pre,
    }
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    dst.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
