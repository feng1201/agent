#!/usr/bin/env bash
set -euo pipefail

# One-click sharded evaluation launcher for eval_lora.py.
#
# How it works:
# - Spawns N processes (N=num_shards). Each process is pinned to ONE GPU via CUDA_VISIBLE_DEVICES.
# - Each process runs eval_lora.py with --num_shards N --shard_id i.
# - After all shards finish, merges results via merge_eval_shards.py.
#
# Requirements:
# - Run inside an srun allocation with enough GPUs, e.g. --gres=gpu:8
# - Use a working py310 interpreter path.
#
# Examples (inside srun --gres=gpu:8 ...):
#   bash scripts/eval_lora_sharded.sh \
#     --python /finance_ML/fengninghui/conda_envs/latentmas_vllm_py310/bin/python \
#     --mode compare \
#     --num_shards 8 \
#     --output_dir /finance_ML/fengninghui/LatentMAS/outputs/eval_compare_sharded \
#     --method latent_mas \
#     --model_name /finance_ML/fengninghui/TSbasemodel/Qwen38btext \
#     --task medqa --split test --prompt sequential \
#     --latent_steps 2 --max_new_tokens 512 --temperature 0 --top_p 1.0 \
#     --generate_bs 1 --max_samples 200 \
#     --lora_path /finance_ML/fengninghui/LatentMAS/outputs/lora_grpo_agent1/final
#
#   # base only:
#   bash scripts/eval_lora_sharded.sh ... --mode single --tag base
#
#   # lora only:
#   bash scripts/eval_lora_sharded.sh ... --mode single --tag lora --lora_path ...

usage() {
  cat <<'EOF'
Usage:
  bash scripts/eval_lora_sharded.sh --python PY --mode {compare|single} --num_shards N --output_dir DIR [more eval_lora.py args]

Required:
  --python       Absolute path to python (py310 env recommended)
  --mode         compare | single
  --num_shards   e.g. 8
  --output_dir   where shard outputs are written

Single-mode extra:
  --tag          base | lora

All remaining args are passed to eval_lora.py.
EOF
}

PYTHON=""
MODE=""
NUM_SHARDS=""
OUTPUT_DIR=""
TAG=""

EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PYTHON="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --num_shards) NUM_SHARDS="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "$PYTHON" || -z "$MODE" || -z "$NUM_SHARDS" || -z "$OUTPUT_DIR" ]]; then
  usage
  exit 2
fi
if [[ "$MODE" != "compare" && "$MODE" != "single" ]]; then
  echo "[ERR] --mode must be compare|single"
  exit 2
fi
if [[ "$MODE" == "single" && -z "$TAG" ]]; then
  echo "[ERR] --mode single requires --tag base|lora"
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_PY="${ROOT_DIR}/eval_lora.py"
MERGE_PY="${ROOT_DIR}/merge_eval_shards.py"

mkdir -p "$OUTPUT_DIR"

echo "[eval_sharded] python=$PYTHON"
echo "[eval_sharded] mode=$MODE num_shards=$NUM_SHARDS output_dir=$OUTPUT_DIR"

PIDS=()
for ((i=0; i<NUM_SHARDS; i++)); do
  # Pin each worker to one GPU. This is data-parallel eval sharding (speedup).
  # Also reduce CPU thread contention.
  (
    export CUDA_VISIBLE_DEVICES="$i"
    export OMP_NUM_THREADS=1
    export TOKENIZERS_PARALLELISM=false
    if [[ "$MODE" == "compare" ]]; then
      "$PYTHON" "$EVAL_PY" \
        --compare \
        --num_shards "$NUM_SHARDS" --shard_id "$i" \
        --output_dir "$OUTPUT_DIR" \
        "${EXTRA_ARGS[@]}"
    else
      if [[ "$TAG" == "base" ]]; then
        "$PYTHON" "$EVAL_PY" \
          --num_shards "$NUM_SHARDS" --shard_id "$i" \
          --output_dir "$OUTPUT_DIR" \
          "${EXTRA_ARGS[@]}"
      elif [[ "$TAG" == "lora" ]]; then
        "$PYTHON" "$EVAL_PY" \
          --use_lora \
          --num_shards "$NUM_SHARDS" --shard_id "$i" \
          --output_dir "$OUTPUT_DIR" \
          "${EXTRA_ARGS[@]}"
      else
        echo "[ERR] --tag must be base|lora"
        exit 2
      fi
    fi
  ) &
  PIDS+=("$!")
  echo "[eval_sharded] launched shard $i pid=${PIDS[-1]}"
done

FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    FAIL=1
  fi
done

if [[ "$FAIL" -ne 0 ]]; then
  echo "[eval_sharded] one or more shards failed"
  exit 1
fi

if [[ "$MODE" == "compare" ]]; then
  "$PYTHON" "$MERGE_PY" --input_dir "$OUTPUT_DIR" --mode compare --num_shards "$NUM_SHARDS"
else
  "$PYTHON" "$MERGE_PY" --input_dir "$OUTPUT_DIR" --mode single --tag "$TAG" --num_shards "$NUM_SHARDS"
fi

echo "[eval_sharded] done"


