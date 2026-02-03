# 纯hf测试
srun --gres=gpu:1 --reservation=finai --cpus-per-task=4 --mem-per-cpu=32G --exclude= --pty /bin/bash
source /finance_ML/fengninghui/miniconda3/etc/profile.d/conda.sh
conda activate /finance_ML/fengninghui/conda_envs/latentmas_vllm_py310
CUDA_VISIBLE_DEVICES=0 python run.py --method latent_mas --model_name /finance_ML/fengninghui/TSbasemodel/Qwen38btext --task medqa --prompt sequential --max_samples 1 --max_new_tokens 512 --latent_steps 2 --device cuda:0
exit

# Phase 1：HF-only GRPO + LoRA 最小训练闭环
cd /finance_ML/fengninghui/LatentMAS
source /finance_ML/fengninghui/miniconda3/etc/profile.d/conda.sh
conda activate /finance_ML/fengninghui/conda_envs/latentmas_vllm_py310

# 确保安装了 peft（requirements.txt 已加）
pip install -r requirements.txt

CUDA_VISIBLE_DEVICES=0 python train_grpo.py \
  --model_name /finance_ML/fengninghui/TSbasemodel/Qwen38btext \
  --task medqa \
  --prompt sequential \
  --device cuda:0 \
  --latent_steps 2 \
  --max_new_tokens 256 \
  --temperature 0.6 \
  --top_p 0.95 \
  --group_size 2 \
  --train_steps 20 \
  --output_dir outputs/lora_grpo_qwen3_8b_medqa

# srun先照抄下面命令
source /finance_ML/fengninghui/miniconda3/etc/profile.d/conda.sh
conda activate /finance_ML/fengninghui/conda_envs/latentmas_vllm_py310
hash -r
which python
python -V

# 绝对路径启动

cd /finance_ML/fengninghui/LatentMAS
CUDA_VISIBLE_DEVICES=0 /finance_ML/fengninghui/conda_envs/latentmas_vllm_py310/bin/python train_grpo.py --model_name /finance_ML/fengninghui/TSbasemodel/Qwen38btext --task medqa --prompt sequential --device cuda:0

# 下面给第一个LLM上lora 证明梯度回传有效

CUDA_VISIBLE_DEVICES=0 python train_agent1_proof.py \
  --model_name /finance_ML/fengninghui/TSbasemodel/Qwen38btext \
  --task medqa \
  --prompt sequential \
  --device cuda:0 \
  --latent_steps 1 \
  --max_steps 3 \
  --output_dir outputs/proof_agent1_grad

上面的不能多卡，新的尝试多卡
CUDA_VISIBLE_DEVICES=0,1 /finance_ML/fengninghui/conda_envs/latentmas_vllm_py310/bin/python \
  /finance_ML/fengninghui/LatentMAS/train_agent1_proof.py \
  --model_name /finance_ML/fengninghui/TSbasemodel/Qwen38btext \
  --task medqa \
  --prompt sequential \
  --device cuda:0 \
  --device_map auto \
  --latent_steps 1 \
  --max_steps 3 \
  --require_disable_adapter

# 绝对路径启动2卡 (同上)
CUDA_VISIBLE_DEVICES=0,1 /finance_ML/fengninghui/conda_envs/latentmas_vllm_py310/bin/python \
  /finance_ML/fengninghui/LatentMAS/train_agent1_proof.py --device_map auto ...

# 8卡训练gpro的第一个LLM # latent_steps可以进行一定的修改
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /finance_ML/fengninghui/conda_envs/latentmas_vllm_py310/bin/python \
  /finance_ML/fengninghui/LatentMAS/train_grpo_agent1.py \
  --model_name /finance_ML/fengninghui/TSbasemodel/Qwen38btext \
  --task medqa \
  --prompt sequential \
  --device cuda:0 \
  --device_map auto \
  --latent_steps 40 \
  --train_steps 25 \
  --max_new_tokens 1024 \
  --group_size 8 \
  --tf_sequential \
  --output_dir /finance_ML/fengninghui/LatentMAS/outputs/lora_grpo_agent1_mp8_v2
s

下面是评估脚本
1) 只评估 base（不加 LoRA）
CUDA_VISIBLE_DEVICES=0 /finance_ML/fengninghui/conda_envs/latentmas_vllm_py310/bin/python \
  /finance_ML/fengninghui/LatentMAS/eval_lora.py \
  --method latent_mas \
  --model_name /finance_ML/fengninghui/TSbasemodel/Qwen38btext \
  --task medqa --split test \
  --max_samples 200 \
  --temperature 0 \
  --output_dir /finance_ML/fengninghui/LatentMAS/outputs/eval_base

2) 只评估 LoRA（加 LoRA）
CUDA_VISIBLE_DEVICES=0,1 /finance_ML/fengninghui/conda_envs/latentmas_vllm_py310/bin/python \
  /finance_ML/fengninghui/LatentMAS/eval_lora.py \
  --method latent_mas \
  --model_name /finance_ML/fengninghui/TSbasemodel/Qwen38btext \
  --task medqa --split test \
  --max_samples 200 \
  --temperature 0 \
  --lora_path /finance_ML/fengninghui/LatentMAS/outputs/lora_grpo_agent1_mp8/final\
  --use_lora \
  --output_dir /finance_ML/fengninghui/LatentMAS/outputs/eval_lora

3) 一键对比（base vs base+LoRA）
CUDA_VISIBLE_DEVICES=0 /finance_ML/fengninghui/conda_envs/latentmas_vllm_py310/bin/python \
  /finance_ML/fengninghui/LatentMAS/eval_lora.py \
  --method latent_mas \
  --model_name /finance_ML/fengninghui/TSbasemodel/Qwen38btext \
  --task medqa --split test \
  --max_samples 200 \
  --temperature 0 \
  --lora_path /finance_ML/fengninghui/LatentMAS/outputs/lora_grpo_agent1/final \
  --compare \
  --output_dir /finance_ML/fengninghui/LatentMAS/outputs/eval_compare

# 之前的太慢了
cd /finance_ML/fengninghui/LatentMAS

bash scripts/eval_lora_sharded.sh \
  --python /finance_ML/fengninghui/conda_envs/latentmas_vllm_py310/bin/python \
  --mode compare \
  --num_shards 8 \
  --output_dir /finance_ML/fengninghui/LatentMAS/outputs/eval_compare_sharded \
  --method latent_mas \
  --model_name /finance_ML/fengninghui/TSbasemodel/Qwen38btext \
  --task medqa --split test --prompt sequential \
  --latent_steps 2 --max_new_tokens 512 --temperature 0 --top_p 1.0 \
  --generate_bs 1 --max_samples 200 \
  --lora_path /finance_ML/fengninghui/LatentMAS/outputs/lora_grpo_agent1/final

默认超参数
method=latent_mas
task=medqa split=test prompt=sequential
latent_steps=2
max_new_tokens=512
temperature=0 top_p=1.0（稳定对比）
generate_bs=1
max_samples=200
num_shards=8

8卡base
cd /finance_ML/fengninghui/LatentMAS
bash scripts/eval_base_medqa_8gpu.sh

8卡lora
cd /finance_ML/fengninghui/LatentMAS
bash scripts/eval_lora_medqa_8gpu.sh

换一个lora
cd /finance_ML/fengninghui/LatentMAS
LORA_PATH=/finance_ML/fengninghui/LatentMAS/outputs/your_new_lora/final bash scripts/eval_lora_medqa_8gpu_v2.sh


./scripts/upload_one_file_github_pat.sh \
 --repo feng1201/agent \
 --local /finance_ML/fengninghui/LatentMAS/test.md \
 --remote test.md \
 --branch main \
 --message "Update test.md "

