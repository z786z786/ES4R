#!/usr/bin/env bash
set -euo pipefail

: "${BLSP_MODEL:?Set BLSP_MODEL to your stage-1 or base checkpoint path}"
: "${DATA_ROOT:?Set DATA_ROOT to processed stage-2 dataset path}"
: "${SAVE_ROOT:?Set SAVE_ROOT to stage-2 output directory}"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-config/dp_config_zero2.json}"

python -m torch.distributed.run \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  train.py \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  --dataset_save_dir "${DATA_ROOT}" \
  --output_dir "${SAVE_ROOT}" \
  --remove_unused_columns False \
  --seed 1 \
  --do_train True \
  --bf16 True \
  --learning_rate 1e-5 \
  --weight_decay 0.05 \
  --max_grad_norm 1.0 \
  --warmup_steps 200 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 8 \
  --blsp_model "${BLSP_MODEL}" \
  --unfreeze_qwen True \
  --unfreeze_whisper False \
  --unfreeze_adapter True \
  --loss_names "response_ce,response_kl" \
  --logging_steps 20 \
  --save_steps 500 \
  --save_total_limit 1
