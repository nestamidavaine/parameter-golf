# Debug Log — Recurrent Core + Learned Feedback + QAT

## Environment Setup

| Component | Value |
|-----------|-------|
| GPU | NVIDIA H200 (143GB HBM) |
| CUDA driver | 13.0 |
| PyTorch | 2.11.0+cu130 |
| Python | 3.12.3 |
| Flash Attention | FA3 Hopper (pre-built wheel from `varunneal/flash-attention-hopper`) |
| OS | Ubuntu 24.04 (noble), kernel 6.11.0-1016-nvidia |

## Setup Steps

### 1. Virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 2. PyTorch installation
Initial attempt with `cu124` failed (CUDA version mismatch: system has 13.0, PyTorch compiled for 12.4).
Reinstalled with nightly `cu128` first, then switched to `flash-attn` which pulled in `cu130`:
```bash
pip install flash-attn --no-build-isolation
# This auto-installed torch 2.11.0+cu130 with CUDA 13.0 bindings
```

### 3. Flash Attention 3 (Hopper)
The model code imports `from flash_attn_interface import flash_attn_func` which is FA3 (Hopper-specific).

**Attempt 1 — Build from source:** Cloned `Dao-AILab/flash-attention`, ran `setup.py install` from `hopper/` directory. Extremely slow: 451 CUDA kernel files, only 42/451 completed in ~34 minutes with `MAX_JOBS=4`. Killed.

**Attempt 2 — Pre-built wheels from HuggingFace:** Downloaded from `varunneal/flash-attention-hopper` (`build/torch210-cxx11-cu130-x86_64-linux/flash_attention_hopper/`). Installed the package into site-packages and created a shim module:
```python
# .venv/lib/python3.12/site-packages/flash_attn_interface.py
from flash_attention_hopper.flash_attn_interface import *
```
**Result:** FA3 import and forward pass verified working.

### 4. Data download
```bash
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
```
Downloaded 10 training shards (~2GB) + 1 validation shard (~124MB) + tokenizer.

## Issues Encountered

### Issue 1: CUDA version mismatch
- **Symptom:** PyTorch cu124 couldn't compile extensions against CUDA 13.0
- **Fix:** Installed PyTorch with cu130 support via flash-attn dependency chain

### Issue 2: `flash_attn_interface` not available as pip package
- **Symptom:** `ModuleNotFoundError: No module named 'flash_attn_interface'`
- **Root cause:** FA3 (Hopper) is a separate build from the `hopper/` directory, not published to PyPI
- **Fix:** Pre-built wheel from HuggingFace + shim module

### Issue 3: `torch.compile(fullgraph=True)` crashes on PyTorch 2.11 nightly
- **Symptom:** `FailOnRecompileLimitHit: Hard failure due to fullgraph=True`
- **Root cause:** The `RecurrentStabilizer.record_pass()` method uses `.item()` and `list.append()` which cause graph breaks. PyTorch nightly (2.11) is stricter about recompilation limits.
- **Fix:** Disabled `torch.compile` entirely (`compiled_model = base_model`). Also set `TORCH_COMPILE_DISABLE=1` in run scripts.

### Issue 4: Shell sandbox failure (Cursor IDE)
- **Symptom:** All shell commands return empty output, 0ms, exit code 0 (even `false`)
- **Impact:** Could not run smoke test or full experiment
- **Workaround:** Created `run_smoke.sh` and `run_full.sh` scripts for manual execution

## Files Modified

| File | Change |
|------|--------|
| `train_bestbase_recurrent_feedback_learned.py` | `torch.compile` → `base_model` (eager mode) |
| `train_bestbase_recurrent_feedback_fixed.py` | Same |
| `train_bestbase_recurrent_qat.py` | Same |
| `train_utils_recurrent.py` | `torch.compile(forward_logits)` → `base_model.forward_logits` |

## Model Verification

Quick forward pass test (successful):
```
PyTorch 2.11.0+cu130, CUDA 13.0, GPU: NVIDIA H200
FA3: OK
Model created: 19,679,297 params
Feedback module: 2,560 params
Stabilizer: OK
Forward pass loss: 6.9405 - ALL OK!
```

## How to Run

### Smoke test (50 steps, ~5 min)
```bash
bash records/track_10min_16mb/2026-03-25_RecurrentCore_LearnedFeedback_QAT/run_smoke.sh
```

### Full experiment (80 min on 1 GPU ≈ 10 min on 8 GPUs)
```bash
bash records/track_10min_16mb/2026-03-25_RecurrentCore_LearnedFeedback_QAT/run_full.sh
```

### Custom duration
```bash
MINUTES=120 SEED=42 bash records/track_10min_16mb/2026-03-25_RecurrentCore_LearnedFeedback_QAT/run_full.sh
```
