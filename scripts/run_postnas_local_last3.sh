#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

set -a
source configs/postnas_local_last3.env
set +a

: "${RUN_ID:=postnas_local_last3}"
: "${DATA_PATH:=./data/datasets/fineweb10B_sp1024}"
: "${TOKENIZER_PATH:=./data/tokenizers/fineweb_1024_bpe.model}"
: "${ITERATIONS:=200}"
: "${TRAIN_BATCH_TOKENS:=8192}"
: "${VAL_LOSS_EVERY:=0}"
: "${VAL_BATCH_SIZE:=8192}"

python3 train_gpt.py
