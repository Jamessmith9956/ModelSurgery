#!/usr/bin/env python3
"""
Smoke-test repeated-layer helpers on a HF causal LM (e.g. Nemotron), then sweep all (i, j)
with 0 <= i <= j < n_layers and save per-pair quality metrics to a JSONL file.

Designed for Google Colab (H100 / multi-GPU). Full grid is O(n_layers^2) forward passes;
use --limit-pairs or --smoke-only for quick runs.

Usage:
  huggingface-cli login   # if the model is gated

  # Full sweep (can take a long time on large n_layers)
  python nemotron_layer_grid_sweep.py \\
    --model-id nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 \\
    --output results_layer_grid.jsonl \\
    --trust-remote-code \\
    --extra-passes 1

  # Quick smoke only
  python nemotron_layer_grid_sweep.py --model-id gpt2 --smoke-only

  # Verify GPU count and disk before loading 120B (no model load)
  python nemotron_layer_grid_sweep.py --check-env --min-gpus 4
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from test_small_model import (
    _get_decoder_stack,
    benchmark_repeated_layers_quality,
    forward_control_no_repeat,
    forward_repeating_layers,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Layer-repeat (i,j) grid sweep for HF causal LMs")
    p.add_argument(
        "--model-id",
        type=str,
        default="gpt2",
        help="Hugging Face model id",
    )
    p.add_argument(
        "--output",
        type=str,
        default="layer_grid_results.jsonl",
        help="JSONL path (one JSON object per line)",
    )
    p.add_argument("--extra-passes", type=int, default=1, help="Extra passes over layers i..j")
    p.add_argument("--trust-remote-code", action="store_true", help="trust_remote_code for from_pretrained")
    p.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=("auto", "bfloat16", "float16", "float32"),
        help="Model weight dtype",
    )
    p.add_argument("--smoke-only", action="store_true", help="Smoke tests only; no grid")
    p.add_argument("--limit-pairs", type=int, default=0, help="Stop after N (i,j) pairs (0 = no limit)")
    p.add_argument("--short-prompts", action="store_true", help="Single short prompt (faster)")
    p.add_argument(
        "--check-env",
        action="store_true",
        help="Print CUDA device count, names, HF cache paths, disk free; exit without loading a model",
    )
    p.add_argument(
        "--min-gpus",
        type=int,
        default=0,
        help="With --check-env, exit with status 1 if torch.cuda.device_count() < this (0 = no check)",
    )
    return p.parse_args()


def _dtype_kw(args: argparse.Namespace) -> Dict[str, Any]:
    if args.dtype == "auto":
        return {}
    m = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    return {"torch_dtype": m[args.dtype]}


def check_environment(args: argparse.Namespace) -> int:
    """Print CUDA / cache / disk info; optional exit 1 if too few GPUs."""
    n = torch.cuda.device_count()
    print(f"torch.cuda.device_count(): {n}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(n):
            print(f"  cuda:{i} {torch.cuda.get_device_name(i)}")
    hf_home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    hub = os.environ.get("HUGGING_FACE_HUB_CACHE") or os.path.join(hf_home, "hub")
    print(f"HF_HOME (default if unset): {hf_home}")
    print(f"HUGGING_FACE_HUB_CACHE (default if unset): {hub}")
    for path in (hf_home, hub):
        try:
            usage = shutil.disk_usage(path)
            free_gb = usage.free / (1024**3)
            print(f"Disk free on {path}: {free_gb:.1f} GiB")
        except OSError as e:
            print(f"Disk usage for {path}: ({e})")
    if args.min_gpus > 0 and n < args.min_gpus:
        print(f"ERROR: need >= {args.min_gpus} GPUs, have {n}", file=sys.stderr)
        return 1
    return 0


def load_model_and_tokenizer(args: argparse.Namespace):
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    kwargs: Dict[str, Any] = dict(trust_remote_code=args.trust_remote_code)
    kwargs.update(_dtype_kw(args))
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"

    if "nvfp4" in args.model_id.lower():
        raise RuntimeError(
            "NVFP4 is not supported for this Transformers load path: checkpoint Linear weights are "
            "packed (e.g. shape [1024, 1344]) but the model is instantiated with full "
            "intermediate_size (e.g. [1024, 2688]).\n\n"
            "For nemotron_layer_grid_sweep.py / manual layer repeats, use a BF16 or FP8 checkpoint, e.g.:\n"
            "  nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16\n"
            "  (or the FP8 model id from NVIDIA on Hugging Face).\n\n"
            "NVFP4 checkpoints are intended for vLLM / sglang (see the model README), not plain "
            "AutoModelForCausalLM.from_pretrained here."
        )

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **kwargs)
    model.eval()
    return model, tok


def run_smoke_tests(
    model: torch.nn.Module,
    tokenizer,
    args: argparse.Namespace,
    prompts: List[str],
) -> None:
    device = next(model.parameters()).device
    text = "Hello"
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    stack = _get_decoder_stack(model)
    n = len(stack.layers)
    print(f"Smoke: n_layers={n}, device={device}")

    with torch.no_grad():
        _ = forward_control_no_repeat(model, input_ids, attention_mask=attention_mask)
        print("  forward_control_no_repeat: OK")

        _ = forward_repeating_layers(
            model,
            input_ids,
            attention_mask=attention_mask,
            repeat_span=(0, min(0, n - 1)),
            extra_passes=args.extra_passes,
        )
        print("  forward_repeating_layers (0,0): OK")

        if n > 1:
            j = min(2, n - 1)
            _ = forward_repeating_layers(
                model,
                input_ids,
                attention_mask=attention_mask,
                repeat_span=(0, j),
                extra_passes=args.extra_passes,
            )
            print(f"  forward_repeating_layers (0,{j}): OK")

    metrics = benchmark_repeated_layers_quality(
        model=model,
        tokenizer=tokenizer,
        model_name=args.model_id,
        i=0,
        j=min(1, n - 1),
        extra_passes=args.extra_passes,
        prompts=prompts,
        trust_remote_code=args.trust_remote_code,
    )
    print(f"  benchmark_repeated_layers_quality sample: {metrics}")


def main() -> int:
    args = parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token

    if args.check_env:
        return check_environment(args)

    prompts: List[str]
    if args.short_prompts:
        prompts = ["The capital of France is"]
    else:
        prompts = [
            "Explain in one sentence what a KV cache is.",
            "The capital of France is",
            "Write a short tagline for a GPU cloud:",
            "2+2=",
        ]

    print(f"Loading {args.model_id} ...")
    t0 = time.perf_counter()
    model, tokenizer = load_model_and_tokenizer(args)
    print(f"Loaded in {time.perf_counter() - t0:.1f}s")

    run_smoke_tests(model, tokenizer, args, prompts)

    if args.smoke_only:
        print("Smoke-only: done.")
        return 0

    stack = _get_decoder_stack(model)
    n_layers = len(stack.layers)
    total_pairs = n_layers * (n_layers + 1) // 2
    print(f"Grid: n_layers={n_layers}, total (i,j) pairs={total_pairs}")
    if args.limit_pairs:
        print(f"Will stop after {args.limit_pairs} pairs")

    meta = {
        "type": "meta",
        "model_id": args.model_id,
        "n_layers": n_layers,
        "extra_passes": args.extra_passes,
        "dtype": args.dtype,
        "short_prompts": args.short_prompts,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(json.dumps(meta) + "\n")

    pairs_done = 0
    for i in range(n_layers):
        for j in range(i, n_layers):
            row: Dict[str, Any] = {
                "type": "result",
                "i": i,
                "j": j,
                "extra_passes": args.extra_passes,
            }
            t1 = time.perf_counter()
            try:
                metrics = benchmark_repeated_layers_quality(
                    model=model,
                    tokenizer=tokenizer,
                    model_name=args.model_id,
                    i=i,
                    j=j,
                    extra_passes=args.extra_passes,
                    prompts=prompts,
                    trust_remote_code=args.trust_remote_code,
                )
                row["ok"] = True
                row["metrics"] = metrics
            except Exception as e:  # noqa: BLE001
                row["ok"] = False
                row["error"] = f"{type(e).__name__}: {e}"
            row["elapsed_s"] = time.perf_counter() - t1
            row["ts"] = datetime.now(timezone.utc).isoformat()

            with open(args.output, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

            pairs_done += 1
            print(f"  ({i},{j}) ok={row.get('ok')} elapsed={row['elapsed_s']:.2f}s")

            if args.limit_pairs and pairs_done >= args.limit_pairs:
                print(f"Stopped after {pairs_done} pairs (--limit-pairs)")
                return 0

    print(f"Done. Wrote {pairs_done} result rows to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
