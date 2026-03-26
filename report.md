# Report — Recurrent Core + Learned Feedback + QAT Experiment

## Overview

This report covers the setup and analysis of the **Recurrent Core with Learned Error-Feedback Correction and Full-Rollout QAT** submission for the OpenAI Parameter Golf challenge (16MB artifact, 10-minute training budget on 8xH100).

### Challenge Context
The current SOTA on the leaderboard is **1.1194 BPB** (LeakyReLU² + Legal TTT + Parallel Muon). This submission builds on that record by adding:
- A shared recurrent core (stem/core/tail architecture)
- STE-based fake quantization during training (full-rollout QAT)
- Learned low-rank error-feedback correction for quantization residuals

## Architecture Summary

### Stem / Core / Tail Partitioning
```
Input → Stem (3 layers) → Core (2 shared layers × 3 passes) → Tail (3 layers) → Output
                ↓                                                      ↑
            skip connections ────────────────────────────────→ consumed
```

- **Total unique layers:** 8 (3 stem + 2 core + 3 tail)
- **Effective depth:** 12 (3 + 2×3 + 3) via weight reuse in the core
- **Model parameters:** 19,679,297 (base) + 2,560 (feedback)

### Error Feedback Module
The learned feedback correction compensates for quantization error amplification across recurrence passes:

```
e_k = U(V^T h_k)           — low-rank residual approximation (rank 2)
c_k = diag(d) · e_k        — diagonal correction (512 params)
h_{k+1} = f_{W_q}(h_k + c_k)  — corrected recurrent update
```

Correction is inactive on pass 0 (no prior quantization residual exists).

### Key Components Preserved from SOTA
| Component | Detail |
|-----------|--------|
| Activation | LeakyReLU(0.5)² |
| BigramHash | 1536 |
| XSA | Last 4 unique layers |
| Partial RoPE | 16/64 dims |
| LayerNorm scaling | 1/√(layer+1) |
| VE128 | Layers 6,7 |
| Weight averaging | EMA(0.997) + SWA(every 50) |
| Export | GPTQ-lite int6 + lzma |
| Optimizer | Parallel Muon |

## Environment

| Component | Value |
|-----------|-------|
| GPU | 1× NVIDIA H200 (143GB HBM) |
| CUDA | 13.0 |
| PyTorch | 2.11.0+cu130 |
| Flash Attention | FA3 Hopper (pre-built) |

## Experiment Status

### What Was Completed
1. **Full environment setup** — PyTorch + FA3 + dependencies + data
2. **Model verification** — Forward pass produces valid loss (6.94 at init)
3. **Code compatibility fixes** — Disabled `torch.compile` for PyTorch nightly compat
4. **Run scripts created** — `run_smoke.sh` (50 steps) and `run_full.sh` (80 min)

### What Remains (Shell Sandbox Failure)
The Cursor IDE shell tool entered a non-functional state during the session, preventing execution of training runs. The scripts are ready to run manually:

```bash
# Smoke test (~5 min)
bash records/track_10min_16mb/2026-03-25_RecurrentCore_LearnedFeedback_QAT/run_smoke.sh

# Full run (80 min on 1 GPU ≈ 10 min on 8 GPUs)
bash records/track_10min_16mb/2026-03-25_RecurrentCore_LearnedFeedback_QAT/run_full.sh
```

## Expected Results

### Script 3 (Learned Feedback) — the strongest variant

Based on the submission README and the architecture design:

1. **QAT alone** (Script 1) should significantly reduce the quantization gap that plagued previous recurrent approaches (PR #363 saw 900× error amplification)

2. **Learned diagonal feedback** (Script 3) should outperform both QAT-only and fixed feedback by:
   - Adapting the correction to the actual error distribution
   - Only adding 2,560 extra parameters (negligible impact on artifact size)
   - Being compatible with the existing GPTQ-lite int6 + lzma export pipeline

3. **Expected BPB range:** If recurrence successfully adds depth without degradation, the model should achieve comparable or better BPB than the base 11-layer record (~1.12-1.13), with the added benefit of fewer unique parameters (8 vs 11 layers).

### Scaling on 1 GPU
The `full_run_1gpu.sh` uses `grad_accum_steps=8` to simulate the 8-GPU batch size. At 80 minutes on 1 GPU, this approximates the 10-minute 8-GPU training budget in terms of optimizer steps completed. The key metrics to watch:

- **val_bpb during training** — Should decrease steadily
- **post-EMA val_bpb** — Should be the best checkpoint quality
- **int6 roundtrip val_bpb** — The official metric (after quantization + compression)
- **sliding window eval** — Typically improves over standard eval by ~0.01-0.02 BPB

## Key Design Questions (Experimental Plan)

| Experiment | Script | Question |
|-----------|--------|----------|
| A | QAT only | Does QAT alone fix recurrence quantization? |
| B | Fixed feedback | Does a tiny correction help beyond QAT? |
| C | Learned feedback | Does learned feedback beat fixed at same budget? |
| D | Learned + TTT | Which TTT regime is safest for shared weights? |
| E | Learned + stabilizers | Do clipping/scaling/Jacobian penalty help? |

Script 3 (learned feedback) is expected to be the best because it can adapt its correction to the actual quantization error distribution during training, while the fixed version uses a static identity or diagonal.

## Recommendations

1. **Run Script 3 first** — It's the main experimental target with the highest expected performance
2. **Compare against QAT-only** — Script 1 provides the ablation baseline
3. **Monitor h_norms and growth_ratios** — The stabilizer diagnostics will show whether recurrence is staying stable
4. **Check int6 roundtrip quality** — The gap between pre/post quantization BPB is the key metric for whether QAT+feedback is working
