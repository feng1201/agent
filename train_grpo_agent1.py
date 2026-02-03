"""
GRPO training for the *first agent* (planner) only, while strictly keeping Judger unchanged.

核心约束（满足你的对比实验需求）：
- Judger forward / generation / logprob 计算阶段：必须禁用 LoRA adapter（使用 peft.disable_adapter()）
- 只更新 LoRA（共享插入到模型线性层），但 Judger 始终不用 LoRA，因此 Judger 行为保持 base 一致

关键技术点：
- 需要可微 latent rollout（past_kv 带计算图），这样 teacher-forcing logprob 的梯度才能回传到 planner 的 LoRA。
- 采样仍然是离散 token（不可微），但策略梯度用 logprob 的梯度即可。

注意：
- 这是 Phase 2-ish 的端到端梯度路线，显存开销比 Phase 1 大很多。
- 建议先用：latent_steps=1、max_new_tokens=64、group_size=2、tf_sequential=True 试跑。
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timedelta
from dataclasses import asdict, dataclass
from typing import List, Optional

import torch
from tqdm.auto import tqdm

from data import load_medqa
from models import ModelWrapper
from prompts import build_agent_message_sequential_latent_mas, build_agent_message_hierarchical_latent_mas
from utils import extract_mcq_choice, normalize_answer, set_seed


def _build_messages(role: str, question: str, *, prompt_style: str, args):
    if prompt_style == "sequential":
        return build_agent_message_sequential_latent_mas(
            role=role, question=question, context="", method="latent_mas", args=args
        )
    return build_agent_message_hierarchical_latent_mas(
        role=role, question=question, context="", method="latent_mas", args=args
    )


def _pad_1d(seqs: List[torch.Tensor], pad_id: int, device: torch.device):
    max_len = max(int(s.numel()) for s in seqs)
    bsz = len(seqs)
    ids = torch.full((bsz, max_len), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((bsz, max_len), dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        s = s.to(device=device, dtype=torch.long)
        L = int(s.numel())
        ids[i, :L] = s
        attn[i, :L] = 1
    return ids, attn


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


def _freeze_non_lora(model) -> None:
    # 更稳妥：把所有非 LoRA 参数冻结，只训练 LoRA。
    for name, p in model.named_parameters():
        if "lora_" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False


def main() -> None:
    if sys.version_info >= (3, 12):
        raise RuntimeError(
            f"检测到 Python {sys.version.split()[0]}（{sys.executable}）。请使用 py310 环境运行。"
        )

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--task", type=str, default="medqa", choices=["medqa"])
    parser.add_argument("--prompt", type=str, default="sequential", choices=["sequential", "hierarchical"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--device_map", type=str, default=None, help="多卡省显存：auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_tqdm", action="store_true", help="关闭进度条 UI（tqdm）。")

    # latent rollout (planner only)
    parser.add_argument("--latent_steps", type=int, default=1)
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--latent_space_realign", action="store_true")

    # sampling
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0, help="采样多样性：top_k>0 会引入额外随机性。")
    parser.add_argument("--typical_p", type=float, default=1.0, help="采样多样性：typical_p<1 可改变分布形状。")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help=">1 可减少重复/模板化输出。")
    parser.add_argument("--group_size", type=int, default=2)
    parser.add_argument(
        "--per_sample_seed",
        action="store_true",
        help="每次采样前重置随机种子（step/k 不同），用于进一步增加 group 内多样性。",
    )

    # train
    parser.add_argument("--train_steps", type=int, default=50)
    parser.add_argument("--max_train_samples", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument(
        "--tf_sequential",
        action="store_true",
        default=True,
        help="默认开启：teacher-forcing 逐样本计算，避免 K 倍 past_kv 扩张 OOM。",
    )

    # LoRA
    parser.add_argument("--lora_init_path", type=str, default=None, help="载入已有 LoRA adapter 继续训练（路径）")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    parser.add_argument(
        "--require_disable_adapter",
        action="store_true",
        default=True,
        help="严格模式：必须能 disable_adapter()，否则报错（保证 Judger 不更新/不受 LoRA 影响）。",
    )

    # output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/finance_ML/fengninghui/LatentMAS/outputs/lora_grpo_agent1",
        help="LoRA 保存目录（按你的要求默认放在 LatentMAS 目录下）",
    )
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument(
        "--debug_dump_generations",
        action="store_true",
        default=True,
        help="将每步的生成结果（clean+raw）追加写入 output_dir/debug_generations.jsonl，便于排查 pred=None。",
    )
    parser.add_argument(
        "--debug_max_chars",
        type=int,
        default=4000,
        help="debug 打印/写文件时，每条输出最多保留多少字符（从末尾截取）。",
    )

    args = parser.parse_args()

    if args.task != "medqa":
        raise ValueError("当前脚本只实现 medqa 的 0/1 reward。")

    set_seed(args.seed)
    device = torch.device(args.device)

    # 为 prompts.py 的断言准备
    args.method = "latent_mas"
    args.text_mas_context_length = -1
    args.device2 = args.device
    args.use_vllm = False
    args.use_second_HF_model = False
    args.enable_prefix_caching = False

    wrapper = ModelWrapper(args.model_name, device, use_vllm=False, args=args)
    # device_map=auto 时，wrapper.device 会被设置成模型第一个参数所在 device
    device = torch.device(wrapper.device)

    try:
        import peft
        from peft import LoraConfig, TaskType, get_peft_model, PeftModel
    except Exception as e:  # pragma: no cover
        raise RuntimeError("缺少 peft 依赖，请先 `pip install peft`") from e

    if args.lora_init_path:
        # 载入已有 adapter 并设为可训练
        wrapper.model = PeftModel.from_pretrained(wrapper.model, args.lora_init_path, is_trainable=True)
    else:
        target_modules = [m.strip() for m in (args.lora_target_modules or "").split(",") if m.strip()]
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        wrapper.model = get_peft_model(wrapper.model, lora_cfg)

    _freeze_non_lora(wrapper.model)
    wrapper.model.train()

    if args.require_disable_adapter and not hasattr(wrapper.model, "disable_adapter"):
        raise RuntimeError("peft 不支持 disable_adapter()，无法严格保证 Judger 不受 LoRA 影响。")

    optimizer = torch.optim.AdamW(
        (p for p in wrapper.model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "train_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    data = list(load_medqa(split="train"))
    random.shuffle(data)
    if args.max_train_samples > 0:
        data = data[: args.max_train_samples]
    if not data:
        raise RuntimeError("Empty training data. Check data/medqa.json.")

    pad_id = int(wrapper.tokenizer.pad_token_id)
    step_time_ema: Optional[float] = None
    ema_beta = 0.9
    global_step = 0
    stats_history: List[TrainStats] = []
    pbar = tqdm(total=int(args.train_steps), disable=bool(args.no_tqdm))

    for step in range(1, int(args.train_steps) + 1):
        t0 = time.time()
        item = data[(step - 1) % len(data)]
        q = item["question"]
        gold = item.get("gold", "")

        # 1) planner 可微 latent rollout -> past_kv（带计算图）
        planner_msgs = _build_messages("planner", q, prompt_style=args.prompt, args=args)
        planner_prompt = wrapper.render_chat(planner_msgs, add_generation_prompt=True)
        if args.think:
            planner_prompt = f"{planner_prompt}<think>"
        penc = wrapper.tokenizer(planner_prompt, return_tensors="pt", add_special_tokens=False)
        planner_ids = penc["input_ids"].to(device)
        planner_mask = penc["attention_mask"].to(device)

        past_kv = wrapper.generate_latent_batch_grad(
            planner_ids,
            attention_mask=planner_mask,
            latent_steps=int(args.latent_steps),
            past_key_values=None,
        )

        # 重要：采样阶段不要复用同一个“可微 past_kv”对象。
        # transformers 的 Cache 可能在 generate 过程中被原地扩展/修改，并在 no_grad 下变得不可微，
        # 从而导致后续 loss 不带 grad（报：does not require grad）。
        past_kv_sampling = wrapper.detach_past_key_values(past_kv)

        # 2) judger prompt tokenization（但 judger 必须 disable adapter）
        judger_msgs = _build_messages("judger", q, prompt_style=args.prompt, args=args)
        judger_prompt = wrapper.render_chat(judger_msgs, add_generation_prompt=True)
        if args.think:
            judger_prompt = f"{judger_prompt}<think>"
        jenc = wrapper.tokenizer(judger_prompt, return_tensors="pt", add_special_tokens=False)
        judger_ids = jenc["input_ids"].to(device)
        judger_mask = jenc["attention_mask"].to(device)
        prompt_len = int(judger_mask[0].sum().item())
        prompt_ids_trim = judger_ids[0, :prompt_len].detach().to("cpu")

        # 3) 采样 K 次（Judger 禁用 adapter）
        K = int(args.group_size)
        completions: List[torch.Tensor] = []
        texts: List[str] = []
        rewards: List[float] = []

        with wrapper.model.disable_adapter():
            for _ in range(K):
                if args.per_sample_seed:
                    # 仅影响采样随机性（确保 group 内更容易分叉）
                    torch.manual_seed(int(args.seed) + int(step) * 1000 + len(texts))
                out = wrapper.generate_text_batch_with_ids(
                    judger_ids,
                    judger_mask,
                    max_new_tokens=int(args.max_new_tokens),
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    top_k=int(args.top_k),
                    typical_p=float(args.typical_p),
                    repetition_penalty=float(args.repetition_penalty),
                    past_key_values=past_kv_sampling,
                )
                gen_ids = out["generated_ids"][0]
                gen_text = out["generated_texts"][0]
                # raw decode（保留 special tokens）用于排查“模型只吐 special token/空白”等情况
                gen_text_raw = wrapper.tokenizer.decode(gen_ids, skip_special_tokens=False)
                pred = normalize_answer(extract_mcq_choice(gen_text))
                r = 1.0 if (pred and gold and pred == gold) else 0.0
                completions.append(gen_ids)
                texts.append(gen_text)
                rewards.append(r)
                # 保存每次采样的 raw 文本，后面统一落盘
                if "texts_raw" not in locals():
                    texts_raw = []
                texts_raw.append(gen_text_raw)

        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        adv = rewards_t - rewards_t.mean()

        # 4) teacher-forcing logprob（Judger 禁用 adapter；但 past_kv 带图，因此梯度回到 planner LoRA）
        optimizer.zero_grad(set_to_none=True)

        losses = []
        with wrapper.model.disable_adapter():
            if args.tf_sequential:
                for k in range(K):
                    full_ids = torch.cat([prompt_ids_trim, completions[k].to("cpu")], dim=0)
                    input_ids = full_ids.unsqueeze(0).to(device)
                    attn = torch.ones_like(input_ids, device=device)
                    prompt_lengths = torch.tensor([prompt_len], dtype=torch.long, device=device)
                    seq_logprobs, token_counts = wrapper.compute_completion_logprobs(
                        input_ids=input_ids,
                        attention_mask=attn,
                        prompt_lengths=prompt_lengths,
                        past_key_values=past_kv,
                    )
                    norm_logprob = (seq_logprobs / token_counts.to(seq_logprobs.dtype)).squeeze(0)
                    losses.append(-(adv[k] * norm_logprob))
                loss = torch.stack(losses).mean()
            else:
                full_seqs = [torch.cat([prompt_ids_trim, c.to("cpu")], dim=0) for c in completions]
                input_ids, attn = _pad_1d(full_seqs, pad_id=pad_id, device=device)
                prompt_lengths = torch.full((K,), prompt_len, dtype=torch.long, device=device)
                # 这里不做 past_kv repeat（太容易 OOM）；强制用 sequential 更稳
                raise RuntimeError("Please use --tf_sequential for agent1 GRPO (memory-safe).")

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
        acc = avg_reward
        avg_gen_len = float(sum(int(c.numel()) for c in completions) / max(1, K))

        step_time = time.time() - t0
        if step_time_ema is None:
            step_time_ema = step_time
        else:
            step_time_ema = ema_beta * step_time_ema + (1 - ema_beta) * step_time
        eta_sec = float(step_time_ema * max(0, int(args.train_steps) - step)) if step_time_ema else 0.0

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

        # if step == 1 or step % 5 == 0:
        (pbar.write if not args.no_tqdm else print)(json.dumps(asdict(st), ensure_ascii=False))
        preds_dbg = [normalize_answer(extract_mcq_choice(t)) for t in texts]
        keep = int(args.debug_max_chars)
        tails = [(t[-keep:].replace("\n", "\\n") if t else "") for t in texts]
        tails_raw = [((texts_raw[i][-keep:]).replace("\n", "\\n") if (i < len(texts_raw) and texts_raw[i]) else "") for i in range(len(texts))]
        (pbar.write if not args.no_tqdm else print)(f"[sample] gold={gold} rewards={rewards} preds={preds_dbg}")
        # for i, tail in enumerate(tails):
        #     (pbar.write if not args.no_tqdm else print)(f"[sample_raw_tail#{i}] {tail}")
        #     # print(f"[sample_raw_tail_with_special#{i}] {tails_raw[i]}")
        #     # 如果 pred=None 且生成长度打满 max_new_tokens，几乎可以判定是“没来得及输出最终答案”
        #     if preds_dbg[i] is None and int(completions[i].numel()) >= int(args.max_new_tokens):
        #         (pbar.write if not args.no_tqdm else print)(
        #             f"[hint] pred=None 且 gen_len==max_new_tokens({args.max_new_tokens})，很可能输出被截断。"
        #             f" 建议把 --max_new_tokens 调大（例如 256/512），或改 Judger prompt 为短答案模式。"
        #         )

        if args.debug_dump_generations:
            try:
                dbg_path = os.path.join(args.output_dir, "debug_generations.jsonl")
                with open(dbg_path, "a", encoding="utf-8") as f:
                    for i in range(len(texts)):
                        f.write(
                            json.dumps(
                                {
                                    "step": step,
                                    "gold": gold,
                                    "reward": rewards[i],
                                    "parsed_pred": preds_dbg[i],
                                    "gen_len": int(completions[i].numel()),
                                    "max_new_tokens": int(args.max_new_tokens),
                                    "text_clean": texts[i],
                                    "text_raw": texts_raw[i] if i < len(texts_raw) else "",
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
            except Exception:
                pass

        if args.save_every > 0 and step % int(args.save_every) == 0:
            save_dir = os.path.join(args.output_dir, f"step_{step}")
            os.makedirs(save_dir, exist_ok=True)
            wrapper.model.save_pretrained(save_dir)
            wrapper.tokenizer.save_pretrained(save_dir)
            with open(os.path.join(save_dir, "train_stats_tail.jsonl"), "w", encoding="utf-8") as f:
                for row in stats_history[-50:]:
                    f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")

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


