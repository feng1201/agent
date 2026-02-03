#!/usr/bin/env bash
set -euo pipefail

# 8-GPU sharded evaluation: LoRA (base + adapter)
#
# Run inside: srun --gres=gpu:8 ...
#
# You MUST set LORA_PATH (or edit default below).

PYTHON="${PYTHON:-/finance_ML/fengninghui/conda_envs/latentmas_vllm_py310/bin/python}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_NAME="${MODEL_NAME:-/finance_ML/fengninghui/TSbasemodel/Qwen38btext}"
LORA_PATH="${LORA_PATH:-/finance_ML/fengninghui/LatentMAS/outputs/lora_grpo_agent1_mp8_v2/final}"

TASK="${TASK:-medqa}"
SPLIT="${SPLIT:-test}"
PROMPT="${PROMPT:-sequential}"

LATENT_STEPS="${LATENT_STEPS:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1.0}"
GENERATE_BS="${GENERATE_BS:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-200}"

OUTPUT_DIR="${OUTPUT_DIR:-/finance_ML/fengninghui/LatentMAS/outputs/lora_grpo_agent1_mp8_v2/final}"

bash "${ROOT_DIR}/scripts/eval_lora_sharded.sh" \
  --python "$PYTHON" \
  --mode single \
  --tag lora \
  --num_shards 8 \
  --output_dir "$OUTPUT_DIR" \
  --method latent_mas \
  --model_name "$MODEL_NAME" \
  --lora_path "$LORA_PATH" \
  --task "$TASK" --split "$SPLIT" --prompt "$PROMPT" \
  --latent_steps "$LATENT_STEPS" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --generate_bs "$GENERATE_BS" \
  --max_samples "$MAX_SAMPLES"


