#!/usr/bin/env bash
set -euo pipefail

: "${BLSP_MODEL:?Set BLSP_MODEL to your checkpoint path}"
: "${INPUT_FILE:?Set INPUT_FILE to the evaluation manifest path}"
: "${OUTPUT_FILE:?Set OUTPUT_FILE to the prediction output path}"

python generate.py \
  --input_file "${INPUT_FILE}" \
  --output_file "${OUTPUT_FILE}" \
  --blsp_model "${BLSP_MODEL}" \
  --input_field dialogue_history \
  --output_field response \
  --use_emotion
