# Hosting for Nemotron layer-repeat / “model surgery” (Transformers)

This doc implements the **host-model-surgery-workflow** plan: a decision framework and operational checklist for running [`nemotron_layer_grid_sweep.py`](../nemotron_layer_grid_sweep.py) on **large HF causal LMs** (e.g. Nemotron 120B BF16/FP8) with **manual superlayer iteration** in Transformers.

## What the workflow needs

- **Multi-GPU** is effectively required for **120B BF16/FP8**: aggregate VRAM plus activation headroom when using `device_map="auto"`.
- **Large disk** for Hugging Face cache + shards (often **400 GB+** comfortable).
- **Do not use NVFP4** for this script: packed weights vs plain `Linear` load (see README and the sweep script guard).

## 1) Pick your priority (no single “best”)

| Priority | Typical choice | When it fits |
|----------|----------------|--------------|
| **Enterprise / predictable** | **Google Cloud A3 (H100)**, or your org’s **AWS p5 / Azure ND H100** | Multi-day jobs, compliance, VPC/quotas, fewer noisy-neighbor surprises |
| **Lowest $/hour / research spikes** | **RunPod** vs **Vast.ai** vs **Lambda Labs** | Hours–days of experiments; you can tolerate occasional preemption or DIY |
| **One full 8×GPU box** | **8×H100 “host node”** style providers (marketplace listings) | Maximum simplicity: one machine, `device_map="auto"` across all GPUs |

**Practical default**

- **Already on a cloud?** Start with **GCP A3 4× or 8× H100** (or org equivalent) for first end-to-end BF16/FP8 load + smoke sweep.
- **Optimizing $/hour?** Shortlist **RunPod** vs **Vast** for **4× or 8× H100**; prefer **RunPod** if you want less DIY clustering friction; **Vast** if lowest cost is paramount and you accept variability.
- **Google Colab** is usually **not suitable** for this 120B Transformers path unless you have a rare **multi-GPU** runtime—**always verify** with the check below.

## 2) Provision: 4–8× H100 (or equivalent) + large disk

- Target **4–8× H100** (or another setup with **comparable aggregate VRAM**) for 120B-class models.
- Attach a **large data volume** (NVMe or fast network disk) and point cache there (next section).
- Ensure **egress/bandwidth** is enough to pull multi-hundred-GB shards once; subsequent runs reuse cache.

## 3) Validate environment before loading 120B

Run **without** loading the model:

```bash
python nemotron_layer_grid_sweep.py --check-env
```

Confirm `torch.cuda.device_count()` matches what you paid for (e.g. 4 or 8). Optional hard check:

```bash
python nemotron_layer_grid_sweep.py --check-env --min-gpus 4
```

Exits with non-zero status if fewer GPUs are visible.

## 4) Cache on a big volume (avoid root FS filling)

Set cache on your large disk, e.g.:

```bash
export HF_HOME=/mnt/hf-cache
export TRANSFORMERS_CACHE=/mnt/hf-cache/hub   # optional; HF_HOME is enough for many setups
```

`--check-env` prints effective paths and disk free space for the current directory and `HF_HOME`.

## 5) Smoke, then chunked sweep

1. **Smoke only** (after GPUs + disk look good):

   ```bash
   python nemotron_layer_grid_sweep.py \
     --model-id nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 \
     --trust-remote-code \
     --smoke-only
   ```

2. **Chunked grid** (debug / partial runs):

   ```bash
   python nemotron_layer_grid_sweep.py \
     --model-id nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 \
     --trust-remote-code \
     --short-prompts \
     --limit-pairs 20 \
     --output layer_grid_chunk.jsonl
   ```

3. Remove `--limit-pairs` only when you intend a full **O(n²)** sweep.

## References (public positioning)

Summaries vary by month; re-check each vendor’s **H100 multi-GPU** SKUs, **interconnect** (NVLink/NVSwitch class), and **disk** options before provisioning.
