"""
Proof script: 证明“Judger loss 的梯度能回传到第一个 agent 的 LoRA 参数”，同时 Judger 参数不更新。

思路（最小证据链）：
1) 单模型 + 多 adapter：
   - base 权重全部冻结
   - adapter 'agent1'：可训练（代表第一个 agent）
   - Judger 前向时禁用 adapter（或切到 frozen adapter），确保 Judger 参数不更新
2) 端到端可微 latent rollout：
   - 使用 ModelWrapper.generate_latent_batch_grad 生成 past_key_values（含计算图）
3) 用 gold 监督构造一个极短 completion（例如 \\boxed{C}），计算 NLL：
   - judger forward：参数冻结，但对输入 past_key_values 可求梯度
   - loss.backward 后，打印 agent1 adapter 的 grad norm > 0 即为证据

注意：这是“证明有效性”的实验脚本，不追求性能；建议先用很小 latent_steps 与很短 completion。
"""

import argparse
import json
import os
import sys
from typing import Dict, List

import torch

from data import load_medqa
from models import ModelWrapper
from prompts import build_agent_message_sequential_latent_mas, build_agent_message_hierarchical_latent_mas
from utils import set_seed


def _get_first_agent_role(prompt_style: str) -> str:
    # default_agents() 的第一个是 planner（当前仓库实现）
    return "planner"


def _build_messages(role: str, question: str, *, prompt_style: str, args) -> List[Dict]:
    if prompt_style == "sequential":
        return build_agent_message_sequential_latent_mas(role=role, question=question, context="", method="latent_mas", args=args)
    return build_agent_message_hierarchical_latent_mas(role=role, question=question, context="", method="latent_mas", args=args)


def _grad_norm(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is None:
            continue
        total += float(p.grad.detach().float().norm().item())
    return total


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
    parser.add_argument(
        "--device_map",
        type=str,
        default=None,
        help="多卡省显存：传 auto 让 HF 自动把模型切到多张 GPU（配合 CUDA_VISIBLE_DEVICES=0,1）。",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--latent_steps", type=int, default=1)
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--latent_space_realign", action="store_true")

    # LoRA
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    parser.add_argument("--max_steps", type=int, default=5, help="跑几步证明就够了（每步一个样本）")
    parser.add_argument("--output_dir", type=str, default="outputs/proof_agent1_grad")
    parser.add_argument(
        "--require_disable_adapter",
        action="store_true",
        help="严格模式：必须能使用 peft 的 disable_adapter() 来保证 Judger 不更新；否则直接报错。",
    )

    args = parser.parse_args()

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

    # base 冻结
    for p in wrapper.model.parameters():
        p.requires_grad = False

    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except Exception as e:  # pragma: no cover
        raise RuntimeError("缺少 peft 依赖，请先 `pip install peft`") from e

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
    wrapper.model.train()

    # 如果 PEFT 支持多 adapter，可命名为 agent1；不支持也没关系（默认 adapter 就是 agent1）
    try:
        wrapper.model.set_adapter("default")
    except Exception:
        pass

    optimizer = torch.optim.AdamW((p for p in wrapper.model.parameters() if p.requires_grad), lr=args.lr)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    data = list(load_medqa(split="train"))
    if not data:
        raise RuntimeError("Empty medqa data. Check data/medqa.json")

    # 只取前 max_steps 条
    for step in range(1, args.max_steps + 1):
        item = data[(step - 1) % len(data)]
        q = item["question"]
        gold = (item.get("gold", "") or "").strip().lower()
        if gold not in ["a", "b", "c", "d"]:
            continue

        # 1) 第一个 agent（planner）可微 latent rollout -> past_kv（含计算图）
        first_role = _get_first_agent_role(args.prompt)
        planner_msgs = _build_messages(first_role, q, prompt_style=args.prompt, args=args)
        planner_prompt = wrapper.render_chat(planner_msgs, add_generation_prompt=True)
        if args.think:
            planner_prompt = f"{planner_prompt}<think>"

        enc = wrapper.tokenizer(
            planner_prompt,
            return_tensors="pt",
            padding=False,
            add_special_tokens=False,
        )
        planner_ids = enc["input_ids"].to(device)
        planner_mask = enc["attention_mask"].to(device)

        past_kv = wrapper.generate_latent_batch_grad(
            planner_ids,
            attention_mask=planner_mask,
            latent_steps=int(args.latent_steps),
            past_key_values=None,
        )

        # 2) Judger prompt（仅 tokenization）
        judger_msgs = _build_messages("judger", q, prompt_style=args.prompt, args=args)
        judger_prompt = wrapper.render_chat(judger_msgs, add_generation_prompt=True)
        if args.think:
            judger_prompt = f"{judger_prompt}<think>"

        jenc = wrapper.tokenizer(
            judger_prompt,
            return_tensors="pt",
            padding=False,
            add_special_tokens=False,
        )
        judger_ids = jenc["input_ids"].to(device)
        judger_mask = jenc["attention_mask"].to(device)
        prompt_len = int(judger_mask[0].sum().item())

        # 3) gold completion：极短监督，便于省显存
        target = f"\\boxed{{{gold.upper()}}}"
        cenc = wrapper.tokenizer(target, return_tensors="pt", padding=False, add_special_tokens=False)
        comp_ids = cenc["input_ids"].to(device)  # [1, Lc]

        full_ids = torch.cat([judger_ids, comp_ids], dim=1)
        full_attn = torch.ones_like(full_ids, device=device)
        prompt_lengths = torch.tensor([prompt_len], dtype=torch.long, device=device)

        # 4) Judger 前向：禁用 adapter（不更新 Judger）；但对 past_kv 求梯度
        disable_ctx = None
        if hasattr(wrapper.model, "disable_adapter"):
            disable_ctx = wrapper.model.disable_adapter()
        elif args.require_disable_adapter:
            raise RuntimeError(
                "当前 peft 不支持 disable_adapter()，无法严格保证 Judger 不更新。\n"
                "请升级 peft，或去掉 --require_disable_adapter（不建议用于对比实验）。"
            )

        optimizer.zero_grad(set_to_none=True)
        if disable_ctx is not None:
            with disable_ctx:
                seq_logprobs, token_counts = wrapper.compute_completion_logprobs(
                    input_ids=full_ids,
                    attention_mask=full_attn,
                    prompt_lengths=prompt_lengths,
                    past_key_values=past_kv,
                )
        else:
            # 退化：不支持 disable_adapter 时，仍可证明梯度回传，但 Judger 也会走同一 adapter。
            seq_logprobs, token_counts = wrapper.compute_completion_logprobs(
                input_ids=full_ids,
                attention_mask=full_attn,
                prompt_lengths=prompt_lengths,
                past_key_values=past_kv,
            )

        loss = -(seq_logprobs / token_counts.to(seq_logprobs.dtype)).mean()
        loss.backward()

        gnorm = _grad_norm((p for p in wrapper.model.parameters() if p.requires_grad))

        optimizer.step()

        print(
            json.dumps(
                {
                    "step": step,
                    "loss": float(loss.detach().item()),
                    "gold": gold,
                    "target": target,
                    "agent1_grad_norm": gnorm,
                    "judger_adapter_disabled": bool(disable_ctx is not None),
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()


