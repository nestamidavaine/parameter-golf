#!/bin/bash
set -euo pipefail

PYTHON="/home/nesta/parameter-golf/.venv/bin/python3"
MINUTES="${MINUTES:-80}"
WALLCLOCK=$((MINUTES * 60))
SEED="${SEED:-1337}"

echo "============================================================"
echo "  Full 1-GPU run: learned feedback variant"
echo "  Wall clock: ${MINUTES} minutes (${WALLCLOCK}s)"
echo "  Seed: ${SEED}"
echo "============================================================"

cd /home/nesta/parameter-golf/records/track_10min_16mb/2026-03-25_RecurrentCore_LearnedFeedback_QAT

PYTHONUNBUFFERED=1 \
TORCH_COMPILE_DISABLE=1 \
DATA_PATH="../../../data/datasets/fineweb10B_sp1024" \
TOKENIZER_PATH="../../../data/tokenizers/fineweb_1024_bpe.model" \
SEED="${SEED}" \
ITERATIONS=20000 \
MAX_WALLCLOCK_SECONDS="${WALLCLOCK}" \
VAL_LOSS_EVERY=2000 \
TRAIN_LOG_EVERY=200 \
WARMUP_STEPS=20 \
WARMDOWN_ITERS=3500 \
TRAIN_BATCH_TOKENS=786432 \
TRAIN_SEQ_LEN=2048 \
EVAL_SEQ_LEN=2048 \
EVAL_STRIDE=64 \
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
MATRIX_LR=0.025 \
SCALAR_LR=0.025 \
TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
MUON_WD=0.04 \
ADAM_WD=0.04 \
GRAD_CLIP_NORM=0.3 \
SWA_ENABLED=1 \
SWA_EVERY=50 \
LATE_QAT=1 \
LATE_QAT_THRESHOLD=0.15 \
TTT_ENABLED=0 \
$PYTHON train_bestbase_recurrent_feedback_learned.py \
    --feedback-mode diagonal --feedback-rank 2 --ttt-regime tail_only

echo ""
echo "Run complete."
