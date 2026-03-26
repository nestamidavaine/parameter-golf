#!/bin/bash
set -euo pipefail

PYTHON="/home/nesta/parameter-golf/.venv/bin/python3"

echo "=== Environment check ==="
$PYTHON -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
from flash_attn_interface import flash_attn_func
print('Flash Attention 3: OK')
"

echo ""
echo "=== Smoke test: Script 3 (learned feedback, 50 steps) ==="
cd /home/nesta/parameter-golf/records/track_10min_16mb/2026-03-25_RecurrentCore_LearnedFeedback_QAT

TORCH_COMPILE_DISABLE=1 \
DATA_PATH="../../../data/datasets/fineweb10B_sp1024" \
TOKENIZER_PATH="../../../data/tokenizers/fineweb_1024_bpe.model" \
ITERATIONS=50 \
MAX_WALLCLOCK_SECONDS=300 \
VAL_LOSS_EVERY=25 \
TRAIN_LOG_EVERY=10 \
WARMUP_STEPS=5 \
WARMDOWN_ITERS=10 \
TRAIN_BATCH_TOKENS=131072 \
TTT_ENABLED=0 \
NUM_STEM_LAYERS=3 \
NUM_CORE_LAYERS=2 \
NUM_TAIL_LAYERS=3 \
NUM_PASSES=3 \
CORE_QUANT_BITS=6 \
CORE_QUANT_ENABLED=1 \
BIGRAM_VOCAB_SIZE=1536 \
XSA_LAST_N=4 \
ROPE_DIMS=16 \
LN_SCALE=1 \
VE_ENABLED=1 \
VE_DIM=128 \
VE_LAYERS="6,7" \
SWA_ENABLED=0 \
$PYTHON train_bestbase_recurrent_feedback_learned.py \
    --feedback-mode diagonal --feedback-rank 2 --ttt-regime tail_only

echo ""
echo "=== Smoke test complete ==="
