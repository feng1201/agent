"""
Phase 1: HF-only GRPO + LoRA (shared parameters across all agents)

目标（最小可行闭环）：
- rollout：复用 LatentMAS(HF) 的 latent rollout -> 得到 Judger 的 past_kv + Judger prompt ids
- 采样：对同一问题采样 K 次 Judger 输出（temperature/top_p），reward=0/1（多选题 A/B/C/D 匹配）
- 更新：teacher-forcing 重新算 Judger completion 的 token logprob（带 past_kv，上下文一致），用最小 GRPO 更新 LoRA

注意：
- 不引入 vLLM；单卡即可（显存不够再调小 max_new_tokens / latent_steps / K）
- 只训练 Judger 输出 token 的 logprob（最易落地）；由于共享参数，更新会间接影响前序 agent 的 latent 行为
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timedelta
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import torch
from tqdm.auto import tqdm

from data import load_medqa
from methods.latent_mas import LatentMASMethod
from models import ModelWrapper
from utils import extract_mcq_choice, normalize_answer, set_seed


def _repeat_past_kv(past_key_values: Optional[Tuple], repeats: int) -> Optional[Tuple]:
    """
    将 batch=1 的 past_key_values 复制成 batch=repeats。
    仅覆盖当前项目 HF-only 路径会出现的 legacy tuple cache 结构：(layer -> (k, v)).
    """
    if past_key_values is None:
        return None
    if repeats == 1:
        return past_key_values
    # transformers 新 Cache 类型兼容
    try:
        from transformers.cache_utils import Cache, DynamicCache  # type: ignore
    except Exception:
        Cache = None  # type: ignore
        DynamicCache = None  # type: ignore

    if Cache is not None and isinstance(past_key_values, Cache):
        legacy = past_key_values.to_legacy_cache()
        repeated_legacy = []
        for layer in legacy:
            k, v = layer
            repeated_legacy.append((k.repeat(repeats, 1, 1, 1), v.repeat(repeats, 1, 1, 1)))
        if DynamicCache is not None:
            return DynamicCache.from_legacy_cache(tuple(repeated_legacy))
        return tuple(repeated_legacy)

    repeated_layers = []
    for layer in past_key_values:
        if isinstance(layer, tuple) and len(layer) == 2 and torch.is_tensor(layer[0]) and torch.is_tensor(layer[1]):
            k, v = layer
            repeated_layers.append((k.repeat(repeats, 1, 1, 1), v.repeat(repeats, 1, 1, 1)))
        else:
            raise TypeError(
                "Unsupported past_key_values structure for repeat. "
                "Expected legacy tuple layers (k, v) tensors."
            )
    return tuple(repeated_layers)


def _pad_1d(seqs: List[torch.Tensor], pad_id: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将一组 1D token 序列 padding 成 [B, L]：
    - input_ids: LongTensor
    - attention_mask: LongTensor（1=有效，0=pad）
    """
    if not seqs:
        raise ValueError("Empty seq list")
    max_len = max(int(s.numel()) for s in seqs)
    batch = len(seqs)
    input_ids = torch.full((batch, max_len), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((batch, max_len), dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        s = s.to(device=device, dtype=torch.long)
        L = int(s.numel())
        if L == 0:
            continue
        input_ids[i, :L] = s
        attn[i, :L] = 1
    return input_ids, attn


@dataclass
class TrainStats:
    step: int
    loss: float
    avg_reward: float
    acc: float
    avg_gen_len: float
    step_time_sec: float
    eta_sec: float


def _fmt_eta(seconds: float) -> str:
    try:
        seconds = float(seconds)
    except Exception:
        return "?"
    if seconds < 0:
        seconds = 0.0
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:d}h{m:02d}m"
    if m > 0:
        return f"{m:d}m{s:02d}s"
    return f"{s:d}s"


def main() -> None:
    # 强约束：本项目当前环境/依赖栈按 python3.10 维护（你们的 env 名也表明如此）。
    # 如果误用 python3.12/3.13，transformers/tokenizers/torch 兼容性更容易出奇怪错误（如 fast tokenizer 解析失败）。
    if sys.version_info >= (3, 12):
        raise RuntimeError(
            f"检测到 Python {sys.version.split()[0]}（{sys.executable}）。\n"
            "请在 srun 会话里先执行：\n"
            "  source /finance_ML/fengninghui/miniconda3/etc/profile.d/conda.sh\n"
            "  conda activate /finance_ML/fengninghui/conda_envs/latentmas_vllm_py310\n"
            "  hash -r\n"
            "然后再运行本脚本。"
        )

    parser = argparse.ArgumentParser()

    # core
    parser.add_argument("--model_name", type=str, required=True, help="HF model id or local path.")
    parser.add_argument("--task", type=str, default="medqa", choices=["medqa"], help="Phase 1 默认先跑 medqa。")
    parser.add_argument("--prompt", type=str, default="sequential", choices=["sequential", "hierarchical"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--device_map",
        type=str,
        default=None,
        help="多卡省显存：传 auto 让 HF 自动把模型切到多张 GPU（配合 CUDA_VISIBLE_DEVICES=0,1 或 0-7）。",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_tqdm", action="store_true", help="关闭进度条 UI（tqdm）。")

    # latent mas
    parser.add_argument("--latent_steps", type=int, default=2)
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--latent_space_realign", action="store_true")
    parser.add_argument("--sequential_info_only", action="store_true")
    parser.add_argument("--latent_only", action="store_true")

    # rollout / sampling
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0, help="采样多样性：top_k>0 会引入额外随机性。")
    parser.add_argument("--typical_p", type=float, default=1.0, help="采样多样性：typical_p<1 可改变分布形状。")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help=">1 可减少重复/模板化输出。")
    parser.add_argument("--group_size", type=int, default=2, help="GRPO group size K（同一问题采样 K 次）")
    parser.add_argument(
        "--per_sample_seed",
        action="store_true",
        help="每次采样前重置随机种子（step/k 不同），用于进一步增加 group 内多样性。",
    )

    # train
    parser.add_argument("--train_steps", type=int, default=50)
    parser.add_argument("--max_train_samples", type=int, default=200, help="从 medqa.json 取多少条做训练。")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="梯度累积（减少峰值显存，代价是更慢）")
    parser.add_argument(
        "--tf_sequential",
        action="store_true",
        help="teacher-forcing logprob 按样本逐个计算（避免 K 倍 past_kv 扩张，显著省显存；但更慢）",
    )

    # lora (peft)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="逗号分隔；留空则用默认（适配 Qwen/Qwen3 系列常见命名）",
    )

    # output
    parser.add_argument("--output_dir", type=str, default="outputs/lora_grpo")
    parser.add_argument("--save_every", type=int, default=50)

    args = parser.parse_args()

    if args.task != "medqa":
        raise ValueError("Phase 1 训练脚本当前只实现了 medqa 的 0/1 reward。")

    set_seed(args.seed)
    device = torch.device(args.device)

    # 训练只走 HF backend；为了复用 prompts.py 的断言，这里把 method 写死为 latent_mas。
    args.method = "latent_mas"
    # latent_mas.py 构造函数里会读 args.device2（虽然 HF-only 不用）
    args.device2 = args.device
    args.use_vllm = False
    args.use_second_HF_model = False
    args.enable_prefix_caching = False
    args.tensor_parallel_size = 1
    args.gpu_memory_utilization = 0.9
    args.vllm_max_num_seqs = 16
    args.vllm_max_model_len = 4096
    args.vllm_enforce_eager = False
    args.text_mas_context_length = -1

    # 1) load model
    wrapper = ModelWrapper(args.model_name, device, use_vllm=False, args=args)
    # device_map=auto 时，wrapper.device 会被设置为模型输入 device（通常为第一张卡）
    device = torch.device(wrapper.device)

    # 2) attach LoRA (PEFT)
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "缺少 peft 依赖。请先 `pip install peft`（本项目 HF-only，不需要 vllm）。"
        ) from e

    target_modules = [m.strip() for m in (args.lora_target_modules or "").split(",") if m.strip()]
    if not target_modules:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    wrapper.model = get_peft_model(wrapper.model, lora_cfg)

    # LoRA 可训练，其余冻结（get_peft_model 默认会处理 requires_grad）
    wrapper.model.train()

    optimizer = torch.optim.AdamW(
        (p for p in wrapper.model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # 3) method (LatentMAS) for building judger context
    method = LatentMASMethod(
        wrapper,
        latent_steps=args.latent_steps,
        judger_max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        generate_bs=1,  # MVP：先固定单条样本（方便 past_kv repeat）
        args=args,
    )

    # 4) dataset
    data = list(load_medqa(split="train"))
    random.shuffle(data)
    if args.max_train_samples > 0:
        data = data[: args.max_train_samples]
    if not data:
        raise RuntimeError("Empty training data. Please check `data/medqa.json`.")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "train_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    # 5) training loop
    pad_id = int(wrapper.tokenizer.pad_token_id)
    stats_history: List[TrainStats] = []
    step_time_ema: Optional[float] = None
    ema_beta = 0.9
    global_step = 0

    pbar = tqdm(total=int(args.train_steps), disable=bool(args.no_tqdm))

    for step in range(1, args.train_steps + 1):
        t0 = time.time()
        item = data[(step - 1) % len(data)]

        # rollout: build judger context (no_grad)
        wrapper.model.eval()
        ctx = method.build_judger_context([item])
        past = ctx["past_key_values"]
        judger_ids = ctx["judger_ids"]  # [1, L]
        judger_mask = ctx["judger_mask"]  # [1, L]

        prompt_len = int(judger_mask[0].sum().item())
        prompt_ids_trim = judger_ids[0, :prompt_len].detach()

        # sample K completions
        K = int(args.group_size)
        completions: List[torch.Tensor] = []
        texts: List[str] = []
        rewards: List[float] = []

        for _ in range(K):
            if args.per_sample_seed:
                torch.manual_seed(int(args.seed) + int(step) * 1000 + len(texts))
            out = wrapper.generate_text_batch_with_ids(
                judger_ids,
                judger_mask,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=int(args.top_k),
                typical_p=float(args.typical_p),
                repetition_penalty=float(args.repetition_penalty),
                past_key_values=past,
            )
            gen_ids = out["generated_ids"][0]
            gen_text = out["generated_texts"][0]
            pred = normalize_answer(extract_mcq_choice(gen_text))
            gold = item.get("gold", "")
            r = 1.0 if (pred and gold and pred == gold) else 0.0

            completions.append(gen_ids)
            texts.append(gen_text)
            rewards.append(r)

        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)  # [K]
        adv = rewards_t - rewards_t.mean()

        # teacher-forcing logprob (with grad)
        wrapper.model.train()
        if args.tf_sequential:
            # 逐个样本计算，避免 batch=K 的 past_kv 复制导致 OOM
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for k in range(K):
                full_ids = torch.cat([prompt_ids_trim.to(device), completions[k].to(device)], dim=0).unsqueeze(0)
                attn1 = torch.ones_like(full_ids, device=device)
                prompt_lengths1 = torch.tensor([prompt_len], dtype=torch.long, device=device)
                seq_logprobs, token_counts = wrapper.compute_completion_logprobs(
                    input_ids=full_ids,
                    attention_mask=attn1,
                    prompt_lengths=prompt_lengths1,
                    past_key_values=past,
                )
                norm_logprob = (seq_logprobs / token_counts.to(seq_logprobs.dtype)).squeeze(0)
                losses.append(-(adv[k] * norm_logprob))
            loss = torch.stack(losses).mean()
            (loss / max(1, int(args.grad_accum_steps))).backward()
        else:
            full_seqs = [torch.cat([prompt_ids_trim.to("cpu"), c], dim=0) for c in completions]
            input_ids, attn = _pad_1d(full_seqs, pad_id=pad_id, device=device)
            prompt_lengths = torch.full((K,), prompt_len, dtype=torch.long, device=device)
            past_rep = _repeat_past_kv(past, repeats=K)

            seq_logprobs, token_counts = wrapper.compute_completion_logprobs(
                input_ids=input_ids,
                attention_mask=attn,
                prompt_lengths=prompt_lengths,
                past_key_values=past_rep,
            )
            norm_logprobs = seq_logprobs / token_counts.to(seq_logprobs.dtype)
            loss = -(adv * norm_logprobs).mean()
            optimizer.zero_grad(set_to_none=True)
            (loss / max(1, int(args.grad_accum_steps))).backward()

        global_step += 1
        if global_step % max(1, int(args.grad_accum_steps)) == 0:
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in wrapper.model.parameters() if p.requires_grad),
                    max_norm=float(args.grad_clip),
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        avg_reward = float(rewards_t.mean().item())
        acc = avg_reward  # 0/1 reward => avg == acc
        avg_gen_len = float(sum(int(c.numel()) for c in completions) / max(1, K))

        step_time = time.time() - t0
        if step_time_ema is None:
            step_time_ema = step_time
        else:
            step_time_ema = ema_beta * step_time_ema + (1 - ema_beta) * step_time
        steps_left = max(0, int(args.train_steps) - step)
        eta_sec = float(step_time_ema * steps_left) if step_time_ema is not None else 0.0

        st = TrainStats(
            step=step,
            loss=float(loss.detach().item()),
            avg_reward=avg_reward,
            acc=acc,
            avg_gen_len=avg_gen_len,
            step_time_sec=float(step_time),
            eta_sec=float(eta_sec),
        )
        stats_history.append(st)
        if not args.no_tqdm:
            finish_at = datetime.now() + timedelta(seconds=float(st.eta_sec))
            pbar.set_description(f"step {step}/{int(args.train_steps)}")
            pbar.set_postfix(
                loss=f"{st.loss:.4f}",
                acc=f"{st.acc:.3f}",
                t=f"{st.step_time_sec:.1f}s",
                eta=_fmt_eta(st.eta_sec),
                end=finish_at.strftime("%H:%M:%S"),
            )
            pbar.update(1)

        if step == 1 or step % 5 == 0:
            (pbar.write if not args.no_tqdm else print)(
                json.dumps(
                    asdict(st),
                    ensure_ascii=False,
                )
            )
            # 只打印 1 条样例，便于确认解析/奖励是否正常
            (pbar.write if not args.no_tqdm else print)(
                f"[sample] gold={item.get('gold')} rewards={rewards} preds={[normalize_answer(extract_mcq_choice(t)) for t in texts]}"
            )

        if args.save_every > 0 and step % int(args.save_every) == 0:
            save_dir = os.path.join(args.output_dir, f"step_{step}")
            os.makedirs(save_dir, exist_ok=True)
            wrapper.model.save_pretrained(save_dir)
            wrapper.tokenizer.save_pretrained(save_dir)
            with open(os.path.join(save_dir, "train_stats_tail.jsonl"), "w", encoding="utf-8") as f:
                for row in stats_history[-50:]:
                    f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")

    # final save
    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    wrapper.model.save_pretrained(final_dir)
    wrapper.tokenizer.save_pretrained(final_dir)
    with open(os.path.join(final_dir, "train_stats.jsonl"), "w", encoding="utf-8") as f:
        for row in stats_history:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")

    if not args.no_tqdm:
        pbar.close()


if __name__ == "__main__":
    main()


