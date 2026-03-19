"""
Minimal test: run repeated-layer quality benchmark on gpt2.
Run from repo root: python test_small_model.py
"""
import inspect
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class _DecoderStack:
    embed_tokens: torch.nn.Module
    layers: torch.nn.ModuleList
    norm: torch.nn.Module
    lm_head: torch.nn.Module


def _first_attr(obj, names: Sequence[str]) -> Optional[torch.nn.Module]:
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return None


def _as_modulelist(x) -> torch.nn.ModuleList:
    if isinstance(x, torch.nn.ModuleList):
        return x
    if isinstance(x, (list, tuple)):
        return torch.nn.ModuleList(list(x))
    raise ValueError(f"Expected ModuleList/list/tuple for layers, got {type(x)}")


def _get_decoder_stack(model: torch.nn.Module) -> _DecoderStack:
    if hasattr(model, "model"):
        inner = model.model
        embed = _first_attr(inner, ("embed_tokens", "tok_embeddings", "wte"))
        norm = _first_attr(inner, ("norm", "ln_f", "final_layernorm", "final_norm"))
        layers_obj = None
        for ln in ("superlayers", "layers", "blocks", "h"):
            if hasattr(inner, ln):
                layers_obj = getattr(inner, ln)
                break
        if embed is not None and norm is not None and layers_obj is not None and hasattr(model, "lm_head"):
            return _DecoderStack(embed_tokens=embed, layers=_as_modulelist(layers_obj), norm=norm, lm_head=model.lm_head)
    if hasattr(model, "transformer"):
        inner = model.transformer
        embed = _first_attr(inner, ("wte", "embed_tokens"))
        norm = _first_attr(inner, ("ln_f", "norm"))
        layers_obj = getattr(inner, "h", None)
        if embed is not None and norm is not None and layers_obj is not None and hasattr(model, "lm_head"):
            return _DecoderStack(embed_tokens=embed, layers=_as_modulelist(layers_obj), norm=norm, lm_head=model.lm_head)
    raise ValueError("Could not locate decoder stack.")


def _call_layer(layer: torch.nn.Module, hidden_states: torch.Tensor, **kwargs):
    sig = inspect.signature(layer.forward)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return layer(hidden_states, **filtered)


def forward_control_no_repeat(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    stack = _get_decoder_stack(model)
    layers = stack.layers
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
    if attention_mask.dim() == 2:
        attention_mask = attention_mask[:, None, None, :].float()  # SDPA expects bool/float
    if position_ids is None:
        pos_mask = attention_mask.squeeze(1).squeeze(1) if attention_mask.dim() == 4 else attention_mask
        position_ids = (pos_mask.cumsum(-1) - 1).clamp(min=0)
    hidden_states = stack.embed_tokens(input_ids)
    layer_kwargs = {
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "use_cache": False,
        "output_attentions": False,
    }
    for layer in layers:
        out = _call_layer(layer, hidden_states, **layer_kwargs)
        hidden_states = out[0] if isinstance(out, (tuple, list)) else out
    hidden_states = stack.norm(hidden_states)
    return stack.lm_head(hidden_states)


def forward_repeating_layers(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    repeat_span: Tuple[int, int],
    extra_passes: int = 1,
) -> torch.Tensor:
    i, j = repeat_span
    if extra_passes < 0:
        raise ValueError("extra_passes must be >= 0")
    stack = _get_decoder_stack(model)
    layers = stack.layers
    n_layers = len(layers)
    if not (0 <= i <= j < n_layers):
        raise ValueError(f"repeat_span must satisfy 0 <= i <= j < {n_layers}. Got {(i, j)}")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
    if attention_mask.dim() == 2:
        attention_mask = attention_mask[:, None, None, :].float()  # SDPA expects bool/float
    if position_ids is None:
        pos_mask = attention_mask.squeeze(1).squeeze(1) if attention_mask.dim() == 4 else attention_mask
        position_ids = (pos_mask.cumsum(-1) - 1).clamp(min=0)
    hidden_states = stack.embed_tokens(input_ids)
    cache_position = torch.arange(input_ids.size(1), device=input_ids.device)
    layer_kwargs = {
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "cache_position": cache_position,
        "use_cache": False,
        "output_attentions": False,
    }
    for idx in range(0, j + 1):
        out = _call_layer(layers[idx], hidden_states, **layer_kwargs)
        hidden_states = out[0] if isinstance(out, (tuple, list)) else out
    for _ in range(extra_passes):
        for idx in range(i, j + 1):
            out = _call_layer(layers[idx], hidden_states, **layer_kwargs)
            hidden_states = out[0] if isinstance(out, (tuple, list)) else out
    for idx in range(j + 1, n_layers):
        out = _call_layer(layers[idx], hidden_states, **layer_kwargs)
        hidden_states = out[0] if isinstance(out, (tuple, list)) else out
    hidden_states = stack.norm(hidden_states)
    return stack.lm_head(hidden_states)


def _next_token_kl(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    p = F.log_softmax(logits_p[:, -1, :], dim=-1)
    q = F.log_softmax(logits_q[:, -1, :], dim=-1)
    return (p.exp() * (p - q)).sum(dim=-1)


def continuation_nll(logits: torch.Tensor, input_ids: torch.Tensor, *, prompt_lens: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    seq_len_minus1 = shift_labels.size(1)
    arange = torch.arange(seq_len_minus1, device=input_ids.device).unsqueeze(0)
    cont_mask = arange >= (prompt_lens.unsqueeze(1) - 1).clamp(min=0)
    per_token_nll = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="none"
    ).view(shift_labels.size())
    masked = per_token_nll * cont_mask
    denom = cont_mask.sum(dim=1).clamp(min=1)
    return masked.sum(dim=1) / denom


def benchmark_repeated_layers_quality(
    *,
    model_name: str = "gpt2",
    device: str | None = None,
    prompts: List[str] | None = None,
    reference_continuation: str = "\nThe answer is:",
    i: int = 2,
    j: int = 4,
    extra_passes: int = 2,
) -> Dict[str, float]:
    if prompts is None:
        prompts = [
            "Explain in one sentence what a KV cache is.",
            "The capital of France is",
            "Write a short tagline for a GPU cloud:",
            "2+2=",
        ]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float32
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    full_texts = [p + reference_continuation for p in prompts]
    enc_full = tokenizer(full_texts, return_tensors="pt", padding=True)
    enc_prompt = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = enc_full["input_ids"].to(device)
    attention_mask = enc_full["attention_mask"].to(device)
    prompt_lens = enc_prompt["attention_mask"].sum(dim=1).to(device)
    model.eval()
    with torch.no_grad():
        logits_control = forward_control_no_repeat(model, input_ids, attention_mask=attention_mask)
        logits_repeat = forward_repeating_layers(
            model, input_ids, attention_mask=attention_mask, repeat_span=(i, j), extra_passes=extra_passes
        )
        kl = _next_token_kl(logits_repeat, logits_control)
        nll_control = continuation_nll(logits_control, input_ids, prompt_lens=prompt_lens)
        nll_repeat = continuation_nll(logits_repeat, input_ids, prompt_lens=prompt_lens)
    return {
        "next_token_kl_mean": kl.mean().item(),
        "next_token_kl_median": kl.median().item(),
        "nll_control_mean": nll_control.mean().item(),
        "nll_repeat_mean": nll_repeat.mean().item(),
        "nll_delta_mean": (nll_repeat - nll_control).mean().item(),
        "batch_size": float(len(prompts)),
    }


if __name__ == "__main__":
    print("Testing repeated-layer quality benchmark on gpt2...")
    results = benchmark_repeated_layers_quality(model_name="gpt2", i=2, j=4, extra_passes=2)
    for k, v in results.items():
        print(f"  {k}: {v}")
    print("Done.")
