#!/usr/bin/env python3
"""
Fast Colab benchmarks: compare **control** vs **layer-repeated** DINOv3 ViT forwards.

**Global:** ImageNet-1k linear head from [dinov3-in1k-probes](https://github.com/yberreby/dinov3-in1k-probes)
(`yberreby/dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe`) evaluated on a subset of
**Imagenette** (10 ImageNet classes). Requires 512×512 inputs to match the probe.

**Dense (default):** **k-NN** over patch embeddings — **no weights trained**. A bank of labeled
patch features is built from the **frozen** backbone (control forward) on a Scene Parse 150
train slice; val patches are classified by neighbor vote using **control** vs **repeat**
features. This measures dense semantics without any readout training.

**Dense (optional `--dense-mode linear_probe`):** Trains a **small linear layer** on patch
features only (ViT still frozen). That is *not* fine-tuning DINOv3, but it *does* fit new
parameters — use k-NN if you want **zero** training.

**Global:** Uses a **pretrained** IN1k linear head (dinov3-in1k-probes) — **no** training.

Install (Colab GPU):
  pip install torch transformers datasets pillow
  pip install git+https://github.com/yberreby/dinov3-in1k-probes.git

Run:
  python dinov3_colab_benchmarks.py

See README.md for HF token if the base model is gated.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from test_dinov3_layer_repeat import (
    forward_vit_control,
    forward_vit_control_sequence,
    forward_vit_repeating_layers,
    forward_vit_repeating_layers_sequence,
)


# Official Imagenette synsets (same order as label 0..9 in frgfm/imagenette)
_IMAGENETTE_SYNSETS: Tuple[str, ...] = (
    "n01440764",
    "n02102040",
    "n02979186",
    "n03000684",
    "n03028021",
    "n03394916",
    "n03417042",
    "n03425413",
    "n03584829",
    "n03761084",
)


def _imagenette_to_imagenet_indices() -> List[int]:
    """Map Imagenette label 0..9 to ImageNet-1k class indices (0..999)."""
    url = "https://s3.amazonaws.com/pytorch/models/imagenet_class_index.json"
    with urllib.request.urlopen(url, timeout=60) as f:
        idx_to_entry = json.loads(f.read().decode())
    syn_to_idx = {entry[0]: int(k) for k, entry in idx_to_entry.items()}
    return [syn_to_idx[s] for s in _IMAGENETTE_SYNSETS]


def _load_models(
    device: torch.device,
    dtype: torch.dtype,
    base_id: str,
    probe_id: str,
    attn_implementation: str,
) -> Tuple[torch.nn.Module, AutoImageProcessor, nn.Module]:
    processor = AutoImageProcessor.from_pretrained(base_id, size={"height": 512, "width": 512})
    model = AutoModel.from_pretrained(
        base_id,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
    ).to(device)
    model.eval()

    try:
        from dinov3_in1k_probes import DINOv3LinearClassificationHead
    except ImportError as e:
        raise ImportError(
            "Install dinov3-in1k-probes: pip install git+https://github.com/yberreby/dinov3-in1k-probes.git"
        ) from e

    probe = DINOv3LinearClassificationHead.from_pretrained(probe_id).to(device)
    probe.eval()
    return model, processor, probe


def _probe_dtype(probe: nn.Module) -> torch.dtype:
    return next(probe.parameters()).dtype


@torch.no_grad()
def run_global_imagenette(
    *,
    device: torch.device,
    model: torch.nn.Module,
    processor: AutoImageProcessor,
    probe: nn.Module,
    label_to_imagenet: List[int],
    repeat_span: Tuple[int, int],
    extra_passes: int,
    max_samples: int,
    split: str = "validation",
) -> Dict[str, float]:
    from datasets import load_dataset

    ds = load_dataset("frgfm/imagenette", "full_size", split=split)
    ds = ds.select(range(min(max_samples, len(ds))))

    correct_c = 0
    correct_r = 0
    n = 0
    for ex in ds:
        img = ex["image"].convert("RGB")
        label = int(ex["label"])
        target_idx = label_to_imagenet[label]
        inputs = processor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device=device, dtype=model.dtype)

        cls_c = forward_vit_control(model, pixel_values)
        cls_r = forward_vit_repeating_layers(
            model, pixel_values, repeat_span=repeat_span, extra_passes=extra_passes
        )
        pd = _probe_dtype(probe)
        logits_c = probe(cls_c.to(pd))
        logits_r = probe(cls_r.to(pd))
        pred_c = logits_c.argmax(dim=-1).item()
        pred_r = logits_r.argmax(dim=-1).item()
        correct_c += int(pred_c == target_idx)
        correct_r += int(pred_r == target_idx)
        n += 1

    return {
        "global_imagenette_top1_control": correct_c / max(n, 1),
        "global_imagenette_top1_repeat": correct_r / max(n, 1),
        "global_imagenette_delta": (correct_r - correct_c) / max(n, 1),
        "global_n_samples": float(n),
    }


def _patch_tokens_from_sequence(
    sequence: torch.Tensor, num_register_tokens: int
) -> torch.Tensor:
    """[B, T, D] -> [B, Hp, Wp, D] patch grid (excludes CLS + registers)."""
    patch = sequence[:, 1 + num_register_tokens :, :]
    b, n, d = patch.shape
    side = int(n**0.5)
    if side * side != n:
        raise ValueError(f"Non-square patch count {n}; expected Hp*Wp.")
    return patch.reshape(b, side, side, d)


def _resize_mask_to_patches(mask_pil: Image.Image, hp: int, wp: int) -> torch.Tensor:
    """PIL mask -> LongTensor [hp, wp] class ids."""
    m = mask_pil.resize((wp, hp), Image.NEAREST)
    arr = np.array(m)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return torch.from_numpy(arr).long()


def _train_dense_head(
    *,
    device: torch.device,
    dim: int,
    num_classes: int,
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    steps: int,
    lr: float,
) -> nn.Module:
    head = nn.Linear(dim, num_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    xf = train_feats.float()
    yl = train_labels.long()
    head.train()
    for _ in range(steps):
        opt.zero_grad()
        logits = head(xf)
        loss = F.cross_entropy(logits, yl)
        loss.backward()
        opt.step()
    head.eval()
    return head


@torch.no_grad()
def _miou_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    *,
    skip_classes: Tuple[int, ...] = (0,),
    ignore_pixels: int = 255,
) -> float:
    """logits [B, C, H, W], targets [B, H, W]. Skips void pixels (default 255) and optional classes (e.g. background 0)."""
    pred = logits.argmax(dim=1)
    valid = targets != ignore_pixels
    pred = pred[valid]
    tgt = targets[valid]
    ious: List[float] = []
    for c in range(num_classes):
        if c in skip_classes:
            continue
        pred_c = pred == c
        tgt_c = tgt == c
        inter = (pred_c & tgt_c).sum().float()
        union = (pred_c | tgt_c).sum().float()
        if union == 0:
            continue
        ious.append((inter / union).item())
    return sum(ious) / max(len(ious), 1)


@torch.no_grad()
def _miou_from_pred_hw(
    pred_hw: torch.Tensor,
    target_hw: torch.Tensor,
    num_classes: int,
    *,
    skip_classes: Tuple[int, ...] = (0,),
    ignore_pixels: int = 255,
) -> float:
    """pred_hw, target_hw: LongTensor [H, W]."""
    valid = target_hw != ignore_pixels
    pred = pred_hw[valid]
    tgt = target_hw[valid]
    ious: List[float] = []
    for c in range(num_classes):
        if c in skip_classes:
            continue
        pred_c = pred == c
        tgt_c = tgt == c
        inter = (pred_c & tgt_c).sum().float()
        union = (pred_c | tgt_c).sum().float()
        if union == 0:
            continue
        ious.append((inter / union).item())
    return sum(ious) / max(len(ious), 1)


def _knn_patch_labels(
    queries: torch.Tensor,
    bank: torch.Tensor,
    bank_y: torch.Tensor,
    *,
    num_classes: int,
    k: int,
    chunk_q: int = 4096,
) -> torch.Tensor:
    """queries [Q, D], bank [N, D], bank_y [N] long -> pred [Q] long."""
    device = queries.device
    bank_n = F.normalize(bank.float(), dim=-1)
    nq = queries.shape[0]
    out = torch.empty(nq, dtype=torch.long, device=device)
    for s in range(0, nq, chunk_q):
        e = min(s + chunk_q, nq)
        q = F.normalize(queries[s:e].float(), dim=-1)
        sim = q @ bank_n.T
        _, idx = sim.topk(min(k, bank_n.shape[0]), dim=-1)
        lbl = bank_y[idx]
        votes = F.one_hot(lbl, num_classes).sum(dim=1).float()
        out[s:e] = votes.argmax(dim=1)
    return out


def run_dense_scene_parse150_knn(
    *,
    device: torch.device,
    model: torch.nn.Module,
    processor: AutoImageProcessor,
    repeat_span: Tuple[int, int],
    extra_passes: int,
    train_samples: int,
    val_samples: int,
    bg_label: int,
    void_label: int,
    knn_k: int,
) -> Dict[str, float]:
    """Scene Parse 150 dense eval with k-NN — no trainable parameters."""
    from datasets import load_dataset

    ds_train = load_dataset("scene_parse_150", "scene_parsing", split=f"train[:{train_samples}]")
    ds_val = load_dataset("scene_parse_150", "scene_parsing", split=f"validation[:{val_samples}]")

    num_reg = int(model.config.num_register_tokens)
    dim = int(model.config.hidden_size)
    num_classes = 151

    bank_feat_list: List[torch.Tensor] = []
    bank_lbl_list: List[torch.Tensor] = []
    for ex in ds_train:
        img = ex["image"].convert("RGB")
        ann = ex["annotation"]
        inputs = processor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device=device, dtype=model.dtype)
        seq = forward_vit_control_sequence(model, pixel_values)
        patches = _patch_tokens_from_sequence(seq, num_reg)[0]
        hp, wp, _ = patches.shape
        mask_t = _resize_mask_to_patches(ann, hp, wp).to(device)
        flat_f = patches.reshape(-1, dim)
        flat_l = mask_t.reshape(-1)
        valid = (flat_l != bg_label) & (flat_l != void_label)
        bank_feat_list.append(flat_f[valid])
        bank_lbl_list.append(flat_l[valid])

    bank_feats = torch.cat(bank_feat_list, dim=0)
    bank_labels = torch.cat(bank_lbl_list, dim=0).clamp(0, num_classes - 1)
    if bank_feats.shape[0] == 0:
        raise RuntimeError("Empty k-NN bank after masking; check bg/void labels.")

    miou_c: List[float] = []
    miou_r: List[float] = []
    for ex in ds_val:
        img = ex["image"].convert("RGB")
        ann = ex["annotation"]
        inputs = processor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device=device, dtype=model.dtype)
        seq_c = forward_vit_control_sequence(model, pixel_values)
        seq_r = forward_vit_repeating_layers_sequence(
            model, pixel_values, repeat_span=repeat_span, extra_passes=extra_passes
        )

        def _eval_seq(seq: torch.Tensor) -> float:
            patches = _patch_tokens_from_sequence(seq, num_reg)
            hp, wp = patches.shape[1], patches.shape[2]
            mask_t = _resize_mask_to_patches(ann, hp, wp).to(device)
            flat_q = patches.reshape(-1, dim)
            pred_flat = _knn_patch_labels(
                flat_q, bank_feats, bank_labels, num_classes=num_classes, k=knn_k
            )
            pred_hw = pred_flat.reshape(hp, wp)
            return _miou_from_pred_hw(
                pred_hw,
                mask_t,
                num_classes,
                skip_classes=(bg_label,),
                ignore_pixels=void_label,
            )

        miou_c.append(_eval_seq(seq_c))
        miou_r.append(_eval_seq(seq_r))

    mean_c = float(sum(miou_c) / max(len(miou_c), 1))
    mean_r = float(sum(miou_r) / max(len(miou_r), 1))
    return {
        "dense_scene_parse150_knn_miou_control": mean_c,
        "dense_scene_parse150_knn_miou_repeat": mean_r,
        "dense_scene_parse150_knn_delta": mean_r - mean_c,
        "dense_knn_k": float(knn_k),
        "dense_bank_patches": float(bank_feats.shape[0]),
    }


def run_dense_scene_parse150_linear(
    *,
    device: torch.device,
    model: torch.nn.Module,
    processor: AutoImageProcessor,
    repeat_span: Tuple[int, int],
    extra_passes: int,
    train_samples: int,
    val_samples: int,
    train_steps: int,
    bg_label: int,
    void_label: int,
) -> Dict[str, float]:
    """Scene Parse 150: train a linear readout on **frozen** patch features (not fine-tuning ViT)."""
    from datasets import load_dataset

    ds_train = load_dataset("scene_parse_150", "scene_parsing", split=f"train[:{train_samples}]")
    ds_val = load_dataset("scene_parse_150", "scene_parsing", split=f"validation[:{val_samples}]")

    num_reg = int(model.config.num_register_tokens)
    dim = int(model.config.hidden_size)
    # 151 logits: class ids 0..150 inclusive (0 = background)
    num_classes = 151

    train_feat_list: List[torch.Tensor] = []
    train_lbl_list: List[torch.Tensor] = []
    for ex in ds_train:
        img = ex["image"].convert("RGB")
        ann = ex["annotation"]
        inputs = processor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device=device, dtype=model.dtype)
        seq = forward_vit_control_sequence(model, pixel_values)
        patches = _patch_tokens_from_sequence(seq, num_reg)[0]  # [Hp, Wp, D]
        hp, wp, _ = patches.shape
        mask_t = _resize_mask_to_patches(ann, hp, wp).to(device)
        flat_f = patches.reshape(-1, dim)
        flat_l = mask_t.reshape(-1)
        valid = (flat_l != bg_label) & (flat_l != void_label)
        train_feat_list.append(flat_f[valid])
        train_lbl_list.append(flat_l[valid])

    train_feats = torch.cat(train_feat_list, dim=0)
    train_labels = torch.cat(train_lbl_list, dim=0).clamp(0, num_classes - 1)
    if train_labels.numel() == 0:
        raise RuntimeError("No training pixels after masking; check bg/void labels for scene_parse_150.")

    head = _train_dense_head(
        device=device,
        dim=dim,
        num_classes=num_classes,
        train_feats=train_feats,
        train_labels=train_labels,
        steps=train_steps,
        lr=1e-3,
    )

    miou_c: List[float] = []
    miou_r: List[float] = []
    for ex in ds_val:
        img = ex["image"].convert("RGB")
        ann = ex["annotation"]
        inputs = processor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device=device, dtype=model.dtype)
        seq_c = forward_vit_control_sequence(model, pixel_values)
        seq_r = forward_vit_repeating_layers_sequence(
            model, pixel_values, repeat_span=repeat_span, extra_passes=extra_passes
        )

        def _eval_one(seq: torch.Tensor) -> float:
            patches = _patch_tokens_from_sequence(seq, num_reg)
            hp, wp = patches.shape[1], patches.shape[2]
            mask_t = _resize_mask_to_patches(ann, hp, wp).to(device)
            logits = (
                head(patches[0].reshape(-1, dim).float())
                .reshape(hp, wp, num_classes)
                .permute(2, 0, 1)
                .unsqueeze(0)
            )
            return _miou_from_logits(
                logits,
                mask_t.unsqueeze(0),
                num_classes,
                skip_classes=(bg_label,),
                ignore_pixels=void_label,
            )

        miou_c.append(_eval_one(seq_c))
        miou_r.append(_eval_one(seq_r))

    mean_c = float(sum(miou_c) / max(len(miou_c), 1))
    mean_r = float(sum(miou_r) / max(len(miou_r), 1))
    return {
        "dense_scene_parse150_linear_miou_control": mean_c,
        "dense_scene_parse150_linear_miou_repeat": mean_r,
        "dense_scene_parse150_linear_delta": mean_r - mean_c,
        "dense_num_classes": float(num_classes),
        "dense_train_patches": float(train_feats.shape[0]),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DINOv3 Colab benchmarks (global + dense)")
    p.add_argument("--base-model", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    p.add_argument(
        "--probe",
        default="yberreby/dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe",
        help="HF repo for DINOv3LinearClassificationHead (dinov3-in1k-probes)",
    )
    p.add_argument("--i", type=int, default=2, help="repeat span start layer")
    p.add_argument("--j", type=int, default=4, help="repeat span end layer")
    p.add_argument("--extra-passes", type=int, default=2)
    p.add_argument("--global-samples", type=int, default=256, help="Imagenette val images (max 3939)")
    p.add_argument(
        "--dense-mode",
        choices=("knn", "linear_probe"),
        default="knn",
        help="knn = no training (default); linear_probe = train linear readout on frozen features only",
    )
    p.add_argument("--dense-knn-k", type=int, default=11, help="k-NN neighbors for dense k-NN mode")
    p.add_argument("--dense-train", type=int, default=96, help="Scene Parse 150 train images (k-NN bank or linear train)")
    p.add_argument("--dense-val", type=int, default=32, help="Scene Parse 150 val images")
    p.add_argument("--dense-steps", type=int, default=200, help="AdamW steps (linear_probe only)")
    p.add_argument(
        "--dense-bg-label",
        type=int,
        default=0,
        help="Background class in Scene Parse masks (excluded from train pixels; skipped in mIoU mean)",
    )
    p.add_argument(
        "--dense-void-label",
        type=int,
        default=255,
        help="Void/dont-care label in masks (excluded from train; masked in mIoU)",
    )
    p.add_argument("--attn-implementation", default="sdpa")
    p.add_argument("--skip-global", action="store_true")
    p.add_argument("--skip-dense", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() and os.environ.get("DINOV3_ALLOW_CPU") != "1":
        print(
            "No CUDA GPU: use Colab GPU or set DINOV3_ALLOW_CPU=1 for slow CPU.",
            file=sys.stderr,
        )
        sys.exit(2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        dtype = torch.float16

    model, processor, probe = _load_models(
        device, dtype, args.base_model, args.probe, args.attn_implementation
    )
    label_map = _imagenette_to_imagenet_indices()

    if not args.skip_global:
        print("Running global (Imagenette + IN1k linear probe)...", flush=True)
        g = run_global_imagenette(
            device=device,
            model=model,
            processor=processor,
            probe=probe,
            label_to_imagenet=label_map,
            repeat_span=(args.i, args.j),
            extra_passes=args.extra_passes,
            max_samples=args.global_samples,
        )
        for k, v in g.items():
            print(f"  {k}: {v}", flush=True)

    if not args.skip_dense:
        if args.dense_mode == "knn":
            print("Running dense (Scene Parse 150, k-NN patches — no training)...", flush=True)
            d = run_dense_scene_parse150_knn(
                device=device,
                model=model,
                processor=processor,
                repeat_span=(args.i, args.j),
                extra_passes=args.extra_passes,
                train_samples=args.dense_train,
                val_samples=args.dense_val,
                bg_label=args.dense_bg_label,
                void_label=args.dense_void_label,
                knn_k=args.dense_knn_k,
            )
        else:
            print("Running dense (Scene Parse 150, linear readout on frozen features)...", flush=True)
            d = run_dense_scene_parse150_linear(
                device=device,
                model=model,
                processor=processor,
                repeat_span=(args.i, args.j),
                extra_passes=args.extra_passes,
                train_samples=args.dense_train,
                val_samples=args.dense_val,
                train_steps=args.dense_steps,
                bg_label=args.dense_bg_label,
                void_label=args.dense_void_label,
            )
        for k, v in d.items():
            print(f"  {k}: {v}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
