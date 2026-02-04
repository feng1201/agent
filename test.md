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
  --output_dir /finance_ML/fengninghui/LatentMAS/outputs/lora_grpo_agent1_mp8_v3
s

# 多 agent（Planner/Critic/Refiner/Judger）GRPO + 可选“按角色 LoRA”微调
# - 通过 `--train_lora_roles` 选择要训练的角色（例如 planner,refiner）
# - Judger 阶段会 disable_adapter()，保证 Judger 不直接使用 LoRA
#
# 2卡 debug 推荐（更快）：小 latent_steps / 小 group_size / 小 train_steps
CUDA_VISIBLE_DEVICES=0,1 /finance_ML/fengninghui/conda_envs/latentmas_vllm_py310/bin/python \
  /finance_ML/fengninghui/LatentMAS/train_grpo_multiagent.py \
  --model_name /finance_ML/fengninghui/TSbasemodel/Qwen38btext \
  --task medqa \
  --prompt sequential \
  --device cuda:0 \
  --device_map auto \
  --latent_steps 2 \
  --train_steps 2 \
  --max_train_samples 2 \
  --max_new_tokens 128 \
  --group_size 2 \
  --tf_sequential \
  --train_lora_roles planner,refiner \
  --output_dir /finance_ML/fengninghui/LatentMAS/outputs/lora_grpo_multiagent_dbg

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

#8卡4个LLM

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /finance_ML/fengninghui/conda_envs/latentmas_vllm_py310/bin/python \
  /finance_ML/fengninghui/LatentMAS/train_grpo_multiagent.py \
  --model_name /finance_ML/fengninghui/TSbasemodel/qwen34b \
  --task medqa \
  --data_file /finance_ML/fengninghui/LatentMAS/data/medqa_like_train_500.json \
  --prompt sequential \
  --device cuda:0 \
  --device_map auto \
  --latent_steps 40 \
  --train_steps 100 \
  --max_train_samples 150 \
  --max_new_tokens 2048 \
  --group_size 8 \
  --tf_sequential \
  --train_lora_roles planner,refiner \
  --output_dir /finance_ML/fengninghui/LatentMAS/outputs/lora_grpo_multiagent_dbg_v2

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /finance_ML/fengninghui/conda_envs/latentmas_vllm_py310/bin/python \
  /finance_ML/fengninghui/LatentMAS/train_grpo_multiagent.py \
  --model_name /finance_ML/fengninghui/TSbasemodel/qwen34b \
  --task medqa \
  --data_file /finance_ML/fengninghui/LatentMAS/data/medqa_like_train_500.json \
  --prompt sequential \
  --device cuda:0 \
  --device_map auto \
  --latent_steps 20 \
  --train_steps 70 \
  --max_train_samples 150 \
  --max_new_tokens 2048 \
  --group_size 8 \
  --tf_sequential \
  --train_lora_roles planner,refiner \
  --output_dir /finance_ML/fengninghui/LatentMAS/outputs/lora_grpo_multiagent_dbg_v2


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /finance_ML/fengninghui/conda_envs/latentmas_vllm_py310/bin/python \
  /finance_ML/fengninghui/LatentMAS/train_grpo_multiagent.py \
  --model_name /finance_ML/fengninghui/TSbasemodel/Qwen38btext \
  --task medqa \
  --data_file /finance_ML/fengninghui/LatentMAS/data/medqa_like_train_500.json \
  --prompt sequential \
  --device cuda:0 \
  --device_map auto \
  --latent_steps 20 \
  --train_steps 40 \
  --max_train_samples 150 \
  --max_new_tokens 2048 \
  --group_size 4 \
  --tf_sequential \
  --per_sample_seed \
  --train_lora_roles planner,critic,refiner \
  --output_dir /finance_ML/fengninghui/LatentMAS/outputs/lora_grpo_multiagent_dbg_v4_123lora_qwen38b

dapo运行
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /finance_ML/fengninghui/conda_envs/latentmas_vllm_py310/bin/python \
  /finance_ML/fengninghui/LatentMAS/train_dapo_multiagent.py \
  --model_name /finance_ML/fengninghui/TSbasemodel/qwen34b \
  --task medqa \
  --data_file /finance_ML/fengninghui/LatentMAS/data/medqa_like_train_500.json \
  --prompt sequential \
  --device cuda:0 \
  --device_map auto \
  --agent_roles planner,critic,refiner,judger \
  --train_lora_roles planner,refiner \
  --latent_steps 20 \
  --max_new_tokens 2048 \
  --group_size 4 \
  --max_group_size 6 \
  --temperature 0.6 --top_p 0.95 --top_k 0 \
  --dapo_epochs 3 \
  --clip_eps_pos 0.2 --clip_eps_neg 0.2 \
  --per_sample_seed \
  --train_steps 40 \
  --max_train_samples 40 \
  --output_dir /finance_ML/fengninghui/LatentMAS/outputs/lora_dapo_multiagent_dbg_v2



# 8卡评估：对比 multi-agent base vs (planner/refiner) LoRA
# 先设置 LORA_PATH 指向训练输出（final 下应有子目录 planner/ refiner/ ...）
# 例如：LORA_PATH=/finance_ML/fengninghui/LatentMAS/outputs/lora_grpo_multiagent_dbg/final
cd /finance_ML/fengninghui/LatentMAS
bash scripts/eval_multiagent_compare_medqa_8gpu.sh

或者
cd /finance_ML/fengninghui/LatentMAS
export LORA_PATH=/finance_ML/fengninghui/LatentMAS/outputs/lora_grpo_multiagent_dbg/final
bash scripts/eval_multiagent_compare_medqa_8gpu.sh

下面是这个8卡脚本的超参数

MODEL_NAME：基座模型路径
DATA_FILE：评估数据路径（默认 /finance_ML/fengninghui/LatentMAS/data/medqa.json）
MAX_SAMPLES：评估条数（越大越慢）
LATENT_STEPS：latent rollout 步数
MAX_NEW_TOKENS：Judger 最大生成长度
TEMPERATURE / TOP_P：评估建议 TEMPERATURE=0 TOP_P=1.0（确定性）
PROMPT：sequential / hierarchical
AGENT_ROLES：默认 planner,critic,refiner,judger（就是“原文 4 agent”）
LORA_ROLES：哪些角色启用 LoRA（默认 planner,refiner）
OUTPUT_DIR：输出目录

如果是8卡跑lora only
cd /finance_ML/fengninghui/LatentMAS
bash scripts/eval_lora_sharded.sh \
  --python /finance_ML/fengninghui/conda_envs/latentmas_vllm_py310/bin/python \
  --eval_py eval_multiagent_lora.py \
  --mode single --tag base \
  --num_shards 8 \
  --output_dir /finance_ML/fengninghui/LatentMAS/outputs/eval_multiagent_base_sharded_qwen34b \
  --model_name /finance_ML/fengninghui/TSbasemodel/qwen34b \
  --task medqa --split test \
  --data_file /finance_ML/fengninghui/LatentMAS/data/medqa.json \
  --prompt sequential \
  --agent_roles planner,critic,refiner,judger \
  --latent_steps 20 \
  --max_new_tokens 2048 \
  --temperature 0.6 --top_p 0.95 \
  --max_samples 200

同一套超参数下跑 LoRA（multi-agent，指定 LoRA 路径与启用角色）：
cd /finance_ML/fengninghui/LatentMAS
bash scripts/eval_lora_sharded.sh \
  --python /finance_ML/fengninghui/conda_envs/latentmas_vllm_py310/bin/python \
  --eval_py eval_multiagent_lora.py \
  --mode single --tag lora \
  --num_shards 8 \
  --output_dir /finance_ML/fengninghui/LatentMAS/outputs/eval_multiagent_lora_sharded_qwen34b \
  --model_name /finance_ML/fengninghui/TSbasemodel/qwen34b \
  --task medqa --split test \
  --data_file /finance_ML/fengninghui/LatentMAS/data/medqa.json \
  --prompt sequential \
  --agent_roles planner,critic,refiner,judger \
  --latent_steps 20 \
  --max_new_tokens 2048 \
  --temperature 0.6 --top_p 0.95 \
  --max_samples 200 \
  --lora_path /finance_ML/fengninghui/LatentMAS/outputs/lora_grpo_multiagent_dbg_v3/final \
  --lora_roles planner,refiner

bash scripts/eval_lora_sharded.sh \
  --python /finance_ML/fengninghui/conda_envs/latentmas_vllm_py310/bin/python \
  --eval_py eval_multiagent_lora.py \
  --mode single --tag lora \
  --num_shards 8 \
  --output_dir /finance_ML/fengninghui/LatentMAS/outputs/eval_multiagent_lora_sharded_qwen34b_dapo \
  --model_name /finance_ML/fengninghui/TSbasemodel/qwen34b \
  --task medqa --split test \
  --data_file /finance_ML/fengninghui/LatentMAS/data/medqa.json \
  --prompt sequential \
  --agent_roles planner,critic,refiner,judger \
  --latent_steps 20 \
  --max_new_tokens 2048 \
  --temperature 0.6 --top_p 0.95 \
  --max_samples 200 \
  --lora_path /finance_ML/fengninghui/LatentMAS/outputs/lora_dapo_multiagent_dbg/final \
  --lora_roles planner,refiner
