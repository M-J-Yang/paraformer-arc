#!/usr/bin/env bash

set -e

cd /root/autodl-tmp/paraformer-arc/FunASR/examples/industrial_data_pretraining/paraformer

VAL_JSONL="../../../data/list/val.jsonl"
OUTPUT_DIR="./outputs/eval_cer"

# 这里先用你当前训练输出目录
# 你也可以改成具体 checkpoint 或导出的模型目录
MODEL_DIR="./outputs"

DEVICE="cuda:0"

python eval_val_cer.py \
  --val-jsonl "${VAL_JSONL}" \
  --model "${MODEL_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}"