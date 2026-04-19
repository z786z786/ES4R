#!/usr/bin/env bash
set -euo pipefail

: "${BLSP_MODEL:?Set BLSP_MODEL to your checkpoint path}"

python chat_demo.py --blsp_model "${BLSP_MODEL}" --use_emotion
