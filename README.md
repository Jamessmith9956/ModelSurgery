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

---

## Nemotron layer grid sweep (CLI)

Script: [`nemotron_layer_grid_sweep.py`](nemotron_layer_grid_sweep.py)

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
