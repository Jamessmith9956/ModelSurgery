"""
DINOv3 ViT-B/16: repeat a span of encoder layers during inference (model surgery probe).

VRAM: ~86M params — weights ~0.35 GB fp32 / ~0.17 GB bf16; typical single-image inference
often lands around ~1–3 GB GPU including framework overhead (resolution and batch size dependent).

Requires transformers with DINOv3 ViT support (e.g. transformers>=5.3).

Critical: RoPE (`position_embeddings`) is computed once from `pixel_values` and must be
reused for every layer call, including extra passes over the repeated span — same as
`DINOv3ViTModel.forward` in Hugging Face.

**Where to run:** Use a **GPU runtime** (e.g. **Google Colab**: Runtime → Change runtime type →
GPU, then `pip install torch transformers pillow` with `transformers>=5.3`). This repo does
not assume a local GPU; a CPU-only or tight-VRAM machine can skip the `__main__` benchmark
and import the helpers from Colab instead.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel


@dataclass
class _Dinov3Stack:
    embeddings: torch.nn.Module
    rope_embeddings: torch.nn.Module
    layers: torch.nn.ModuleList
    norm: torch.nn.Module


def _as_modulelist(x) -> torch.nn.ModuleList:
    if isinstance(x, torch.nn.ModuleList):
        return x
    if isinstance(x, (list, tuple)):
        return torch.nn.ModuleList(list(x))
    raise ValueError(f"Expected ModuleList/list/tuple for layers, got {type(x)}")


def _get_dinov3_stack(model: torch.nn.Module) -> _Dinov3Stack:
    """Locate DINOv3 ViT stack: embeddings, rope, layer list, final norm."""
    if not hasattr(model, "layer") or not hasattr(model, "embeddings"):
        raise ValueError(
            "Expected a DINOv3 ViT–style model with `.embeddings`, `.rope_embeddings`, `.layer`, `.norm`."
        )
    rope = getattr(model, "rope_embeddings", None)
    norm = getattr(model, "norm", None)
    if rope is None or norm is None:
        raise ValueError("Could not locate rope_embeddings or norm on model.")
    return _Dinov3Stack(
        embeddings=model.embeddings,
        rope_embeddings=rope,
        layers=_as_modulelist(model.layer),
        norm=norm,
    )


def _call_layer(layer: torch.nn.Module, hidden_states: torch.Tensor, **kwargs):
    sig = inspect.signature(layer.forward)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return layer(hidden_states, **filtered)


def forward_vit_control(
    model: torch.nn.Module,
    pixel_values: torch.Tensor,
    *,
    bool_masked_pos: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Match `DINOv3ViTModel.forward`: embeddings → layers (with shared RoPE) → norm → CLS pooler."""
    stack = _get_dinov3_stack(model)
    pixel_values = pixel_values.to(stack.embeddings.patch_embeddings.weight.dtype)
    hidden_states = stack.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)
    position_embeddings = stack.rope_embeddings(pixel_values)
    layer_kw = {"position_embeddings": position_embeddings}
    for layer in stack.layers:
        hidden_states = _call_layer(layer, hidden_states, **layer_kw)
    sequence_output = stack.norm(hidden_states)
    return sequence_output[:, 0, :]


def forward_vit_control_sequence(
    model: torch.nn.Module,
    pixel_values: torch.Tensor,
    *,
    bool_masked_pos: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Full normalized sequence (CLS + register + patch tokens); same as `last_hidden_state` from AutoModel."""
    stack = _get_dinov3_stack(model)
    pixel_values = pixel_values.to(stack.embeddings.patch_embeddings.weight.dtype)
    hidden_states = stack.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)
    position_embeddings = stack.rope_embeddings(pixel_values)
    layer_kw = {"position_embeddings": position_embeddings}
    for layer in stack.layers:
        hidden_states = _call_layer(layer, hidden_states, **layer_kw)
    return stack.norm(hidden_states)


def forward_vit_repeating_layers(
    model: torch.nn.Module,
    pixel_values: torch.Tensor,
    *,
    bool_masked_pos: Optional[torch.Tensor] = None,
    repeat_span: Tuple[int, int],
    extra_passes: int = 1,
) -> torch.Tensor:
    """
    Run layers 0..j, then `extra_passes` additional full passes over layers i..j,
    then layers j+1..L-1; then norm and return CLS embedding (post-norm).
    """
    i, j = repeat_span
    if extra_passes < 0:
        raise ValueError("extra_passes must be >= 0")
    stack = _get_dinov3_stack(model)
    layers = stack.layers
    n_layers = len(layers)
    if not (0 <= i <= j < n_layers):
        raise ValueError(f"repeat_span must satisfy 0 <= i <= j < {n_layers}. Got {(i, j)}")

    pixel_values = pixel_values.to(stack.embeddings.patch_embeddings.weight.dtype)
    hidden_states = stack.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)
    position_embeddings = stack.rope_embeddings(pixel_values)
    layer_kw = {"position_embeddings": position_embeddings}

    for idx in range(0, j + 1):
        hidden_states = _call_layer(layers[idx], hidden_states, **layer_kw)
    for _ in range(extra_passes):
        for idx in range(i, j + 1):
            hidden_states = _call_layer(layers[idx], hidden_states, **layer_kw)
    for idx in range(j + 1, n_layers):
        hidden_states = _call_layer(layers[idx], hidden_states, **layer_kw)

    sequence_output = stack.norm(hidden_states)
    return sequence_output[:, 0, :]


def forward_vit_repeating_layers_sequence(
    model: torch.nn.Module,
    pixel_values: torch.Tensor,
    *,
    bool_masked_pos: Optional[torch.Tensor] = None,
    repeat_span: Tuple[int, int],
    extra_passes: int = 1,
) -> torch.Tensor:
    """Full normalized sequence after span-repeated layer schedule."""
    i, j = repeat_span
    if extra_passes < 0:
        raise ValueError("extra_passes must be >= 0")
    stack = _get_dinov3_stack(model)
    layers = stack.layers
    n_layers = len(layers)
    if not (0 <= i <= j < n_layers):
        raise ValueError(f"repeat_span must satisfy 0 <= i <= j < {n_layers}. Got {(i, j)}")

    pixel_values = pixel_values.to(stack.embeddings.patch_embeddings.weight.dtype)
    hidden_states = stack.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)
    position_embeddings = stack.rope_embeddings(pixel_values)
    layer_kw = {"position_embeddings": position_embeddings}

    for idx in range(0, j + 1):
        hidden_states = _call_layer(layers[idx], hidden_states, **layer_kw)
    for _ in range(extra_passes):
        for idx in range(i, j + 1):
            hidden_states = _call_layer(layers[idx], hidden_states, **layer_kw)
    for idx in range(j + 1, n_layers):
        hidden_states = _call_layer(layers[idx], hidden_states, **layer_kw)

    return stack.norm(hidden_states)


def assert_control_matches_automodel(
    model: torch.nn.Module,
    pixel_values: torch.Tensor,
    *,
    bool_masked_pos: Optional[torch.Tensor] = None,
    rtol: float = 1e-3,
    atol: float = 1e-4,
) -> None:
    """Sanity check: manual control path matches `model(pixel_values).pooler_output`."""
    model.eval()
    device = pixel_values.device
    dtype = pixel_values.dtype
    with torch.no_grad():
        ref = model(pixel_values, bool_masked_pos=bool_masked_pos).pooler_output
        manual = forward_vit_control(model, pixel_values, bool_masked_pos=bool_masked_pos)
    if not torch.allclose(ref, manual, rtol=rtol, atol=atol):
        max_diff = (ref - manual).abs().max().item()
        raise AssertionError(
            f"Control forward does not match AutoModel (max abs diff {max_diff}). "
            f"Check dtype/device and transformers version."
        )


def benchmark_repeated_layers_embedding_drift(
    *,
    model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
    model: Optional[torch.nn.Module] = None,
    processor=None,
    image_urls: Optional[List[str]] = None,
    device: str | None = None,
    i: int = 2,
    j: int = 4,
    extra_passes: int = 2,
    attn_implementation: str = "sdpa",
) -> Dict[str, float]:
    """
    Compare CLS embeddings (post-norm) between control forward and span-repeated forward.
    Metrics: mean 1 - cosine similarity, and mean L2 distance between pooled vectors.
    """
    if image_urls is None:
        image_urls = [
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
            "http://images.cocodataset.org/val2017/000000039769.jpg",
        ]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if processor is None:
        processor = AutoImageProcessor.from_pretrained(model_name)
    if model is None:
        dtype = torch.float32
        if device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
        ).to(device)
    model.eval()

    from transformers.image_utils import load_image

    images = [load_image(url) for url in image_urls]
    inputs = processor(images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    if pixel_values.dtype != next(model.parameters()).dtype:
        pixel_values = pixel_values.to(dtype=next(model.parameters()).dtype)

    with torch.no_grad():
        assert_control_matches_automodel(model, pixel_values)
        pooled_control = forward_vit_control(model, pixel_values)
        pooled_repeat = forward_vit_repeating_layers(
            model, pixel_values, repeat_span=(i, j), extra_passes=extra_passes
        )

    cos_sim = F.cosine_similarity(pooled_control, pooled_repeat, dim=-1)
    drift = 1.0 - cos_sim
    l2 = (pooled_control - pooled_repeat).norm(dim=-1)

    return {
        "cosine_drift_mean": drift.mean().item(),
        "cosine_drift_median": drift.median().item(),
        "l2_mean": l2.mean().item(),
        "l2_median": l2.median().item(),
        "batch_size": float(len(image_urls)),
        "num_layers": float(len(_get_dinov3_stack(model).layers)),
    }


if __name__ == "__main__":
    import os
    import sys

    if not torch.cuda.is_available() and os.environ.get("DINOV3_ALLOW_CPU") != "1":
        print(
            "No CUDA GPU: run this script on a GPU runtime (recommended: Google Colab with a "
            "GPU, then `pip install torch transformers pillow`). See README.md — DINOv3 section. "
            "To force CPU anyway (slow), set DINOV3_ALLOW_CPU=1.",
            file=sys.stderr,
        )
        sys.exit(2)

    print("DINOv3 ViT-B/16 layer-repeat embedding drift (vs control forward)...")
    results = benchmark_repeated_layers_embedding_drift(
        model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
        i=2,
        j=4,
        extra_passes=2,
    )
    for k, v in results.items():
        print(f"  {k}: {v}")
    print("Done.")
