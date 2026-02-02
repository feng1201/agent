# 进入项目目录

cd /finance_ML/fengninghui/LatentMAS

# 申请两张 GPU
srun --gres=gpu:2 --reservation=finai --cpus-per-task=4 --mem-per-cpu=32G --exclude= --pty /bin/bash

# 在 GPU 会话里：激活环境 + 设置 PATH
source /finance_ML/fengninghui/miniconda3/etc/profile.d/conda.sh
conda activate /finance_ML/fengninghui/conda_envs/latentmas_vllm_py310

# 可选：避免 dump_bash_state 的 “command not found”
export PATH=/finance_ML/fengninghui/bin:$PATH


# 运行 LatentMAS with vLLM
CUDA_VISIBLE_DEVICES=0,1 python run.py \
  --method latent_mas \
  --model_name /finance_ML/fengninghui/TSbasemodel/Qwen38btext \
  --task medqa \
  --prompt sequential \
  --split test \
  --max_samples 1 \ # -1为全量运行
  --generate_bs 1 \
  --max_new_tokens 512 \
  --latent_steps 2 \
  --temperature 0.6 \
  --top_p 0.95 \
  --use_vllm \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.80 \
  --vllm_max_num_seqs 8 \
  --vllm_max_model_len 2048 \
  --vllm_enforce_eager \
  --device cuda:0 \
  --device2 cuda:1