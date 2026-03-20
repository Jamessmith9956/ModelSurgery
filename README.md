# ModelSurgery

Replicating work by David Ng https://dnhkng.github.io/posts/rys/

This repo contains a Jupyter notebook (`vllm_cookbook.ipynb`) for running NVIDIA Nemotron with vLLM, plus utilities for **repeating decoder layers** during inference and benchmarking (KL/NLL probes and objective MCQ accuracy on GPQA, ARC-Challenge, HellaSwag).

---

## Running in Google Colab

1. **Open the notebook in Colab**
   - Upload `vllm_cookbook.ipynb` to Google Drive (or clone this repo), then open it with **Open in Colab** (right‑click → Open with → Google Colaboratory), or
   - From Colab: **File → Upload notebook** and select `vllm_cookbook.ipynb`.

2. **Runtime and GPU**
   - **Runtime → Change runtime type → Hardware accelerator: GPU** (T4 or better for vLLM/Nemotron; for the small-model tests, CPU is fine).
   - For the **repeated-layer benchmarks and MCQ evals**, a CPU or a single GPU is enough if you use the default `gpt2` examples.

3. **Install dependencies**
   - Run the notebook cells in order. The first code cells use `%pip install` for vLLM, PyTorch, and Transformers. Let them finish before running model or benchmark cells.
   - For **only** the repeated-layer and benchmark sections (no vLLM), you can run a single install cell:
     ```python
     !pip install -q torch transformers datasets
     ```

4. **What to run**
   - **vLLM + Nemotron**: run from the top through “Load the model” and “Generate responses” (requires a GPU with enough VRAM for the chosen model).
   - **Repeated-layer helpers**: run the “Repeat a span of layers during execution” code cell to define `forward_repeating_layers`, `generate_batched_with_repeated_layers`, and `_get_decoder_stack`.
   - **Quality benchmark (KL/NLL)**: run the “Benchmark: repeated layers vs control (quality-only)” code cell; the example uses `gpt2` by default.
   - **Objective MCQ benchmarks**: run the dataset/loaders cells, then the “run_objective_mcq_benchmarks” cell; again the example uses `gpt2` unless you pass a different `model_name` or an existing `model`/`tokenizer`.

5. **Using your own model (e.g. Nemotron)**
   - Load the Hugging Face model and tokenizer once (e.g. `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` and `AutoTokenizer.from_pretrained(...)`), then pass `model=` and `tokenizer=` into `benchmark_repeated_layers_quality(...)` and `run_objective_mcq_benchmarks(...)` so the notebook does not re-download the model.

6. **DINOv3 ViT-B/16 layer-repeat test (`test_dinov3_layer_repeat.py`)**
   - Intended for a **GPU** (e.g. Colab **T4** or better). Upload or clone the repo, enable a GPU runtime, then:
     ```bash
     pip install torch transformers pillow
     ```
     Use **`transformers>=5.3`** so `DINOv3ViTModel` is available. If the checkpoint is gated, log in (`huggingface-cli login` or Colab secrets).
   - From the repo root:
     ```bash
     python test_dinov3_layer_repeat.py
     ```
   - The script exits early if **no CUDA GPU** is present (so you are not surprised on a CPU-only box). Set **`DINOV3_ALLOW_CPU=1`** to override (slow). Import `forward_vit_control`, `benchmark_repeated_layers_embedding_drift`, etc. from a Colab cell if you prefer not to use `__main__`.

7. **DINOv3 objective benchmarks (`dinov3_colab_benchmarks.py`)**
   - **Global:** [dinov3-in1k-probes](https://github.com/yberreby/dinov3-in1k-probes) **pretrained** IN1k linear head on **Imagenette** at **512×512** — **no training**; only compare control vs layer-repeat **forward** + same frozen probe.
   - **Dense (default `knn`):** **Scene Parse 150** patch **k-NN** — build a labeled feature bank with the **control** forward (frozen ViT), then classify val patches by neighbor vote using **control** vs **repeat** embeddings. **No weights trained** (only inference).
   - **Dense (optional `linear_probe`):** trains a **small linear readout** on frozen patch features (still **not** fine-tuning DINOv3). Use `--dense-mode linear_probe` if you want that SSL-style linear probe.
   - Install:
     ```bash
     pip install torch transformers datasets pillow numpy
     pip install git+https://github.com/yberreby/dinov3-in1k-probes.git
     ```
     Requires **`transformers>=5.3`**. First run downloads Imagenette + Scene Parse 150.
   - Run: `python dinov3_colab_benchmarks.py` (same CUDA guard; `--skip-global` / `--skip-dense` for one track; `--dense-mode knn` or `linear_probe`).

---

## Local / venv

- Create a venv, activate it, then:
  ```bash
  pip install torch transformers datasets
  ```
  (Add `vllm` and version pins from the notebook if you run the full vLLM/Nemotron stack.)
- **Small-model test (no notebook):**
  ```bash
  python test_small_model.py
  ```
  This runs the repeated-layer quality benchmark on `gpt2` and prints KL/NLL metrics.
- **DINOv3 layer-repeat:** see [Running in Google Colab](#running-in-google-colab) §6 — use Colab or another **GPU** machine; `test_dinov3_layer_repeat.py` refuses to run the default benchmark without CUDA.
- **DINOv3 global + dense probes (Colab):** see §7 — [`dinov3_colab_benchmarks.py`](dinov3_colab_benchmarks.py) (Imagenette + IN1k linear probe; Scene Parse 150 patch mIoU).

---

## Nemotron layer grid sweep (CLI)

Script: [`nemotron_layer_grid_sweep.py`](nemotron_layer_grid_sweep.py)

**NVFP4 checkpoints are not supported here** (plain `AutoModelForCausalLM.from_pretrained` fails: packed weight shapes vs full `Linear` dims). Use **BF16** or **FP8** Nemotron ids for this script; use **vLLM** for NVFP4 per the NVIDIA model README.

1. **Smoke test only** (checks `forward_control_no_repeat`, `forward_repeating_layers`, and one `benchmark_repeated_layers_quality` call):
   ```bash
   python nemotron_layer_grid_sweep.py --model-id nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 --trust-remote-code --smoke-only
   ```
   Use `--model-id gpt2` for a fast local check.

2. **Full grid** over all `(i, j)` with `0 <= i <= j < n_layers` (writes JSONL: first line `meta`, then one `result` per pair):
   ```bash
   python nemotron_layer_grid_sweep.py \
     --model-id nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 \
     --trust-remote-code \
     --output layer_grid_results.jsonl \
     --extra-passes 1
   ```
   - Set `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) if the model is gated.
   - On GPU, weights use `device_map="auto"` (multi-GPU friendly).
   - **`--short-prompts`**: one short prompt (much faster on huge models).
   - **`--limit-pairs N`**: stop after N pairs (debugging).
   - Full sweep is **O(n_layers²)**; large models can take a long time.

3. **Google Colab**: upload or clone the repo, install deps (`pip install torch transformers`), then run the same commands in a cell with `!python nemotron_layer_grid_sweep.py ...`. Use an **H100 / A100** runtime and enough disk for the checkpoint; **G4** is usually too small for 120B weights.

---

## Hosting (120B layer-repeat / model surgery)

For **Nemotron 120B**-class runs you typically need **multi-GPU**, **large disk** for the HF cache, and **BF16/FP8** (not NVFP4) on this Transformers path.

- **Full guide**: [`docs/HOSTING.md`](docs/HOSTING.md) — pick enterprise vs marketplace vs 8×GPU box, provision disk, then operational checklist.
- **Verify GPUs and disk before `from_pretrained`**:

  ```bash
  python nemotron_layer_grid_sweep.py --check-env
  python nemotron_layer_grid_sweep.py --check-env --min-gpus 4
  ```

- **Then** smoke → chunked sweep: `--smoke-only`, then `--short-prompts` with `--limit-pairs` (see Nemotron layer grid sweep section above).
