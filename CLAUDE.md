# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

FluxRT is a real-time **stream editing** pipeline: it transforms a webcam/video feed frame-by-frame using **FLUX.2-Klein** (an *instruct image-editing* model, not plain text-to-image) with live prompt and reference-image updates, targeting 15–50 FPS at 512×512 on consumer NVIDIA GPUs. The current input frame is fed as a conditioning image, so this is fundamentally img2img/editing — not generation from scratch.

The package lives under `src/fluxrt/` (installed as the `fluxrt` package via `pyproject.toml`, `package-dir = src`).

## Setup & run

Models are **not** in the repo — they must be cloned via `git lfs` before anything runs (see README "Download Models"):
- `FLUX.2-klein-4B/` (required, full bf16)
- `RIFE-safetensors/flownet.safetensors` (required, frame interpolation)
- `FLUX.2-klein-4B-int8/` (optional, only for `--int8`; the full bf16 model is still required alongside it)
- `LivePortrait/` + `LivePortrait-code/` (optional lip transfer)

Install (Windows / Linux respectively):
```bash
scripts/install.bat        # or: scripts/install-uv.bat
sh scripts/install.sh
```
Manual: Python 3.12, install torch+torchvision for CUDA 12.8 first, then `pip install -r requirements.txt && pip install -e .`. `pyproject.toml` deps are intentionally minimal; `requirements.txt` is the real dependency set.

Entry-point scripts (all in `scripts/`, run with the env active):
```bash
python scripts/run_gui.py                  # PySide6 GUI + virtual webcam (+ Spout on Windows)
python scripts/run_gradio_demo.py          # web UI at http://127.0.0.1:7860/
python scripts/run_cv2_demo.py             # minimal OpenCV demo
python scripts/run_cv2_reference_demo.py   # with reference image
python scripts/run_cv2_paint.py            # iterative paint app
python scripts/process_local_video.py --input in.mp4 --output out.mp4 --prompt "..."
python scripts/run_benchmark.py [--save]   # FPS/latency/VRAM report -> benchmark.md
```
Every script accepts `--int8` to force int8 quantization (equivalent to `enable_int8_quantization: true` in the config).

There is **no test suite**. `run_benchmark.py` is the canonical way to validate that a change still runs end-to-end and to measure its performance impact; share/update `benchmark.md` for perf-relevant changes.

## Configuration

Behavior is driven by JSON configs in `configs/` (each script hard-codes which one it loads). Key fields: `models_path`, `int8_models_path`, `resolution` (**height/width must be divisible by 16** = `vae_scale_factor * 2`), `default_steps` (typically **2** — Klein is step-distilled), `interpolation_exp` (RIFE factor; output batch = `2**interpolation_exp`), `enable_spatial_cache`, `compile_models`, `target_fps`, `use_reference_image` (+ `reference_image_path`/`reference_image_resolution`), `use_lora`/`lora_weights_path`, `mask_calculation_method` (`"auto"` or `"manual"`), `lip_transfer`.

## Architecture

### Three-process design (shared-memory, near-zero-copy)
`StreamProcessor` (`stream_processor/stream_processor.py`) is the public API. It spawns two subprocesses (start method forced to `"spawn"`) and wires them together with `SharedTensor` (`utils/shared_tensor.py`, numpy views over `multiprocessing.shared_memory`):

- **Main process** — owns I/O + UI. Writes camera frames into the input `SharedTensor`, reads the output `SharedTensor`. Sends live updates (prompt, seed, reference image, mask, lip toggle) via a `multiprocessing.Manager().Queue()` command queue.
- **`ModelInferenceSubprocess`** — loads all models, runs the generation loop. Reads input frame → runs the Flux2 pipeline → optional LivePortrait post-process → RIFE interpolation → writes a *batch* of `2**interpolation_exp` frames into the output-batch `SharedTensor`.
- **`OutputSchedulerSubprocess`** — drains the output batch into the single-frame output tensor, pacing playback by `last_processing_time / batch_size` so interpolated frames stream out smoothly.

Coordination is via shared `Value` flags (`pack_is_ready`, `last_processing_time`, `frame_written`), not pipes.

### Flux2 pipeline (`stream_processor/pipeline.py`, `transformer_flux2.py`)
`Flux2KleinPipeline` and `Flux2Transformer2DModel` are **forked-and-modified copies of `diffusers`' Flux2 code** (note the "Copied from diffusers..." comments). Do not assume upstream diffusers behavior — the local copies are patched for caching and to remove Device-to-Host syncs (e.g. precomputed latent H/W). Text encoder is **Qwen3** (`Qwen3ForCausalLM` + `Qwen2TokenizerFast`), using hidden states from layers `(9, 18, 27)`. VAE is `AutoencoderKLFlux2`. Prompts are pre-encoded once into `prompt_embeds` and reused across frames; changing the prompt re-encodes and resets the cache.

### Spatial KV Cache — the core real-time optimization
The speed comes mostly from **reusing computation across adjacent frames**, not from few steps alone. Two cooperating pieces:

1. **`UpdateController`** (`update_controller.py`) produces a per-token **mask** each frame with values `{0, 1, 2}`:
   - `0` = skip (reuse cached K/V/output, compute nothing)
   - `1` = execute only (recompute, don't update cache)
   - `2` = execute **and** update cache
   In `"auto"` mode it diffs the blurred current frame against a cached frame, thresholds + dilates the difference, and max-pools down by `compression_ratio` (16) to mark which spatial regions changed. Text tokens and reference-image tokens get their own mask segments (full recompute once, then `0` until invalidated). The mask layout is `[text | latent | latent | reference?]`.
2. **`SpatialCache`** (in `transformer_flux2.py`) stores K/V (and outputs) **per token, per layer, per timestep** (one `SpatialCache` object keyed by integer timestep, held in `pipeline.spatial_cache`). The transformer forward is patched so FFN, linear projections, query computation, and attention only run on active tokens; cached values fill the rest. README reports only **20–50% of tokens** recomputed per frame.

When editing the transformer, preserve the mask contract and the per-timestep cache keying. `mask_calculation_method: "manual"` lets the app push masks directly (used by the paint app via `StreamProcessor.set_mask`).

### Other components
- **RIFE interpolation** (`interpolation_model.py`, `IFNet`) runs in fp16 and generates `2**interpolation_exp − 1` in-between frames per real frame.
- **int8 path** (`quantized_flux2.py`, `load_quantized_models`) uses `optimum.quanto` `requantize` with a saved `quanto_qmap.json`; only the transformer + text encoder are quantized, VAE stays bf16.
- **Postprocessors** (`postprocessors/`) — `BasePostProcessor` interface; `LivePortraitPostProcessor` transfers webcam facial expressions onto the generated frame (off by default, toggled at runtime).
- **`torch.compile`** is applied to transformer, VAE, and RIFE when `compile_models: true` (first frames are slow while compiling).

### Windows specifics
`run_gui.py` supports **Spout** (in via configurable name, out as constant `FluxRTOutput`) through `SpoutGL`; `triton-windows` is required for model compilation. Virtual webcam output uses `pyvirtualcam` and requires OBS (Windows) or v4l2loopback (Linux).
