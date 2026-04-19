#!/usr/bin/env bash
set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to the processed dual-path dataset path}"
: "${SAVE_ROOT:?Set SAVE_ROOT to the training output directory}"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-config/dp_config_zero2.json}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-8}"
LOSS_NAMES="${LOSS_NAMES:-response_ce,response_kl}"

MODEL_ARGS=()
if [[ -n "${BLSP_MODEL:-}" ]]; then
  MODEL_ARGS+=(--blsp_model "${BLSP_MODEL}")
else
  : "${QWEN_PATH:?Set QWEN_PATH when BLSP_MODEL is not provided}"
  : "${WHISPER_PATH:?Set WHISPER_PATH when BLSP_MODEL is not provided}"
  MODEL_ARGS+=(
    --qwen_model "${QWEN_PATH}"
    --whisper_model "${WHISPER_PATH}"
  )
fi

python3 -m torch.distributed.run \
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
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay 0.05 \
  --max_grad_norm 1.0 \
  --warmup_steps "${WARMUP_STEPS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --unfreeze_qwen True \
  --unfreeze_whisper False \
  --unfreeze_adapter True \
  --loss_names "${LOSS_NAMES}" \
  --logging_steps 20 \
  --save_steps 500 \
  --save_total_limit 1 \
  "${MODEL_ARGS[@]}"
