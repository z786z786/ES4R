#!/usr/bin/env bash
set -euo pipefail

: "${QWEN_PATH:?Set QWEN_PATH to your Qwen checkpoint path}"
: "${WHISPER_PATH:?Set WHISPER_PATH to your Whisper checkpoint path}"
: "${DATA_ROOT:?Set DATA_ROOT to processed stage-1 dataset path}"
: "${SAVE_ROOT:?Set SAVE_ROOT to stage-1 output directory}"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-config/dp_config_zero1.json}"

python -m torch.distributed.run --nproc_per_node="${NPROC_PER_NODE}" train.py \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  --dataset_save_dir "${DATA_ROOT}" \
  --output_dir "${SAVE_ROOT}" \
  --remove_unused_columns False \
  --seed 1 \
  --do_train True \
  --bf16 True \
  --learning_rate 1e-4 \
  --weight_decay 0.05 \
  --max_grad_norm 1.0 \
  --warmup_steps 1000 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 24 \
  --num_train_epochs 1 \
  --whisper_model "${WHISPER_PATH}" \
  --qwen_model "${QWEN_PATH}" \
  --unfreeze_adapter True \
  --loss_names response_kl \
  --logging_steps 20 \
  --save_steps 200 \
  --save_total_limit 1
