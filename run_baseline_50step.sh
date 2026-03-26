#!/bin/bash
set -uo pipefail

PYTHON="/home/nesta/parameter-golf/.venv/bin/python3"
SCRIPT_DIR="/home/nesta/parameter-golf/records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon"
cd "$SCRIPT_DIR"

set -a; source /home/nesta/parameter-golf/.env; set +a

export PYTHONUNBUFFERED=1
export TORCH_COMPILE_DISABLE=1
export DATA_PATH="../../../data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="../../../data/tokenizers/fineweb_1024_bpe.model"
export SEED=1337
export ITERATIONS=50
export MAX_WALLCLOCK_SECONDS=900
export VAL_LOSS_EVERY=25
export TRAIN_LOG_EVERY=10
export WARMUP_STEPS=5
export WARMDOWN_ITERS=10
export TRAIN_BATCH_TOKENS=786432
export TRAIN_SEQ_LEN=2048
export EVAL_SEQ_LEN=2048
export EVAL_STRIDE=0
export NUM_LAYERS=11
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export BIGRAM_VOCAB_SIZE=1536
export XSA_LAST_N=4
export ROPE_DIMS=16
export LN_SCALE=1
export VE_ENABLED=1
export VE_DIM=128
export VE_LAYERS="9,10"
export MATRIX_LR=0.025
export SCALAR_LR=0.025
export TIED_EMBED_LR=0.035
export MUON_MOMENTUM=0.99
export MUON_MOMENTUM_WARMUP_START=0.92
export MUON_MOMENTUM_WARMUP_STEPS=5
export MUON_WD=0.04
export ADAM_WD=0.04
export GRAD_CLIP_NORM=0.3
export SWA_ENABLED=0
export TTT_ENABLED=0

LOG="/home/nesta/parameter-golf/baseline_50step.log"
echo "START baseline SOTA 50-step ($(date +%H:%M:%S))"

$PYTHON -c "
import subprocess, sys, os, re, time

os.environ['WANDB_PROJECT'] = 'parameter-golf'
os.environ['WANDB_NAME'] = 'baseline_SOTA_50step'

import wandb
wandb.init(
    project='parameter-golf',
    name='baseline_SOTA_50step',
    config={
        'method': 'baseline_SOTA',
        'num_layers': 11, 'model_dim': 512, 'num_heads': 8,
        'num_passes': 1, 'recurrence': False,
        'train_batch_tokens': 786432, 'train_seq_len': 2048,
        'iterations': 50, 'seed': 1337,
    },
)

proc = subprocess.Popen(
    [sys.executable, 'train_gpt.py'],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, bufsize=1,
)

logf = open('$LOG', 'w')
for line in proc.stdout:
    logf.write(line)
    logf.flush()
    print(line, end='')

    m = re.search(r'step:(\d+)/\d+ val_loss:([\d.]+) val_bpb:([\d.]+) train_time:(\d+)ms step_avg:([\d.]+)ms', line)
    if m:
        step = int(m.group(1))
        wandb.log({'val_loss': float(m.group(2)), 'val_bpb': float(m.group(3)), 'step_avg_ms': float(m.group(5))}, step=step)

    m = re.search(r'step:(\d+)/\d+ train_loss:([\d.]+) train_time:(\d+)ms step_avg:([\d.]+)ms', line)
    if m:
        step = int(m.group(1))
        wandb.log({'train_loss': float(m.group(2)), 'step_avg_ms': float(m.group(4))}, step=step)

proc.wait()
logf.close()
wandb.finish()
print(f'EXIT CODE: {proc.returncode}')
" 2>&1

echo "DONE baseline ($(date +%H:%M:%S))"
