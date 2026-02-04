"""
DAPO-style training for multi-agent LatentMAS (HF-only).

Implements two key ideas from DAPO ("Decoupled Clip and Dynamic sAmpling Policy Optimization"):
1) Dynamic Sampling:
   - For each prompt, keep sampling until the group has reward variance (or reaches a cap).
   - If still no learning signal (all rewards equal / all parsed_pred None), skip the step.

2) Decoupled Clip (PPO-like objective with different clip ranges for adv>0 vs adv<0),
   and multiple policy update epochs per rollout batch (so ratios move away from 1).

Design constraints (aligned with this repo):
- Multi-agent latent rollout (planner/critic/refiner -> judger) with KV-cache communication.
- Judger is kept unchanged by disabling adapters during sampling and logprob computation.
- Gradients flow through the KV cache into selected trainable role adapters (per-role LoRA).

Notes:
- This is still a simplified implementation aimed for correctness/engineering, not an exact re-implementation
  of every optimization detail in the DAPO system paper.
- It is expensive: each "epoch" recomputes the full latent rollout to keep gradients consistent.
"""

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from datasets import load_dataset

from models import ModelWrapper, _past_length
from prompts import build_agent_message_hierarchical_latent_mas, build_agent_message_sequential_latent_mas
from utils import extract_mcq_choice, normalize_answer, set_seed

try:
    from transformers.cache_utils import Cache  # type: ignore
except Exception:
    Cache = None  # type: ignore


@dataclass
class TrainStats:
    step: int
    skipped: bool
    skip_reason: str
    loss: float
    avg_reward: float
    reward_std: float
    avg_gen_len: float
    samples_used: int
    epochs: int
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


def _parse_csv_roles(text: str) -> List[str]:
    items = [x.strip() for x in (text or "").split(",")]
    return [x for x in items if x]


def _build_messages(role: str, question: str, *, prompt_style: str, args) -> List[Dict]:
    if prompt_style == "sequential":
        return build_agent_message_sequential_latent_mas(
            role=role, question=question, context="", method="latent_mas", args=args
        )
    return build_agent_message_hierarchical_latent_mas(
        role=role, question=question, context="", method="latent_mas", args=args
    )


def _slice_last_tokens(t: torch.Tensor, tokens_to_keep: int) -> torch.Tensor:
    if tokens_to_keep <= 0:
        return t[..., 0:0, :].contiguous()
    keep = min(int(tokens_to_keep), int(t.shape[-2]))
    start = int(t.shape[-2]) - keep
    return t[..., start:, :].contiguous()


def _truncate_past_kv(past_kv: Optional[Tuple], tokens_to_keep: int) -> Optional[Tuple]:
    """
    Keep only the last `tokens_to_keep` cached positions for each layer.
    Similar to LatentMASMethod._truncate_past, but avoids detach (keeps gradients for the kept slice).
    """
    if past_kv is None or tokens_to_keep <= 0:
        return None
    if Cache is not None and isinstance(past_kv, Cache):
        legacy = past_kv.to_legacy_cache()
        trimmed_legacy = tuple(tuple(_slice_last_tokens(t, tokens_to_keep) for t in layer) for layer in legacy)
        return past_kv.__class__.from_legacy_cache(trimmed_legacy)
    trimmed_layers = []
    for layer in past_kv:
        if isinstance(layer, tuple):
            trimmed_layers.append(tuple(_slice_last_tokens(t, tokens_to_keep) for t in layer))
        elif torch.is_tensor(layer):
            trimmed_layers.append(_slice_last_tokens(layer, tokens_to_keep))
        else:
            trimmed_layers.append(layer)
    return tuple(trimmed_layers)


def _freeze_non_lora_and_select_trainable_adapters(model, trainable_roles: Sequence[str]) -> None:
    # Freeze all by default
    for _, p in model.named_parameters():
        p.requires_grad = False
    # Enable LoRA params for selected adapters
    trainable_roles = list(trainable_roles)
    for name, p in model.named_parameters():
        if "lora_" not in name:
            continue
        if any(f".{role}." in name for role in trainable_roles):
            p.requires_grad = True


def _ensure_adapters(model, trainable_roles: Sequence[str], lora_cfg) -> None:
    existing = set(getattr(model, "peft_config", {}).keys()) if hasattr(model, "peft_config") else set()
    for role in trainable_roles:
        if role in existing:
            continue
        model.add_adapter(role, lora_cfg)


def _save_role_adapters(model, *, save_root: str, roles: Sequence[str]) -> None:
    roles = list(roles)
    if not roles:
        return
    for role in roles:
        out_dir = os.path.join(save_root, role)
        os.makedirs(out_dir, exist_ok=True)
        try:
            model.save_pretrained(out_dir, selected_adapters=[role])
        except TypeError:
            try:
                model.set_adapter(role)
            except Exception:
                pass
            model.save_pretrained(out_dir)


def _load_medqa_items_from_file(data_file: str) -> List[Dict]:
    ds = load_dataset("json", data_files=data_file, split="train")
    choice_map = {"0": "A", "1": "B", "2": "C", "3": "D"}
    rows: List[Dict] = []
    for item in ds:
        question = item["query"]
        raw_answer = str(item.get("answer", ""))
        answer = ""
        for idx, op in enumerate(item.get("options", [])):
            if raw_answer and raw_answer in str(op):
                answer = choice_map.get(str(idx), "")
                break
        gold = normalize_answer(answer)
        rows.append({"question": question, "solution": answer, "gold": gold})
    return rows


def _compute_old_logprob_norm(
    *,
    wrapper: ModelWrapper,
    past_kv_detached: Optional[Tuple],
    prompt_ids_trim: torch.Tensor,
    prompt_len: int,
    completion_ids: torch.Tensor,
    device: torch.device,
) -> float:
    full_ids = torch.cat([prompt_ids_trim, completion_ids.to("cpu")], dim=0).unsqueeze(0).to(device)
    attn = torch.ones_like(full_ids, device=device)
    prompt_lengths = torch.tensor([prompt_len], dtype=torch.long, device=device)
    with torch.no_grad():
        seq_logprobs, token_counts = wrapper.compute_completion_logprobs(
            input_ids=full_ids,
            attention_mask=attn,
            prompt_lengths=prompt_lengths,
            past_key_values=past_kv_detached,
        )
        val = (seq_logprobs / token_counts.to(seq_logprobs.dtype)).squeeze(0).detach().float().item()
    return float(val)


def _decoupled_clip_ratio(ratio: torch.Tensor, adv: torch.Tensor, eps_pos: float, eps_neg: float) -> torch.Tensor:
    eps_pos = float(eps_pos)
    eps_neg = float(eps_neg)
    eps = torch.where(adv >= 0, torch.as_tensor(eps_pos, device=ratio.device, dtype=ratio.dtype), torch.as_tensor(eps_neg, device=ratio.device, dtype=ratio.dtype))
    lo = 1.0 - eps
    hi = 1.0 + eps
    return torch.clamp(ratio, lo, hi)


def main() -> None:
    if sys.version_info >= (3, 12):
        raise RuntimeError(f"检测到 Python {sys.version.split()[0]}（{sys.executable}）。请使用 py310 环境运行。")

    p = argparse.ArgumentParser()
    # core
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--task", type=str, default="medqa", choices=["medqa"])
    p.add_argument("--data_file", type=str, default="/finance_ML/fengninghui/LatentMAS/data/medqa.json")
    p.add_argument("--prompt", type=str, default="sequential", choices=["sequential", "hierarchical"])
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--device_map", type=str, default=None, help="多卡省显存：auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_tqdm", action="store_true")

    # agents
    p.add_argument("--agent_roles", type=str, default="planner,critic,refiner,judger")
    p.add_argument("--train_lora_roles", type=str, default="planner", help="启用并训练的角色（逗号分隔）。")

    # latent rollout
    p.add_argument("--latent_steps", type=int, default=2)
    p.add_argument("--think", action="store_true")
    p.add_argument("--latent_space_realign", action="store_true")
    p.add_argument("--sequential_info_only", action="store_true")
    p.add_argument("--latent_only", action="store_true")

    # sampling (judger)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--typical_p", type=float, default=1.0)
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument("--group_size", type=int, default=4, help="initial group size K")
    p.add_argument("--per_sample_seed", action="store_true")

    # dynamic sampling (DAPO)
    p.add_argument("--dynamic_sampling", action="store_true", default=True)
    p.add_argument("--max_group_size", type=int, default=16, help="dynamic sampling upper bound")
    p.add_argument("--min_reward_std", type=float, default=1e-6, help="if reward std <= this, treat as no signal")
    p.add_argument("--min_unique_preds", type=int, default=2, help="require at least this many unique parsed preds to stop sampling")

    # DAPO/PPO update
    p.add_argument("--dapo_epochs", type=int, default=2, help="policy update epochs per rollout batch")
    p.add_argument("--clip_eps_pos", type=float, default=0.2, help="clip eps for adv>=0")
    p.add_argument("--clip_eps_neg", type=float, default=0.2, help="clip eps for adv<0")
    p.add_argument("--adv_norm", action="store_true", default=True, help="normalize advantages by std within group")

    # train
    p.add_argument("--train_steps", type=int, default=50)
    p.add_argument("--max_train_samples", type=int, default=200)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--require_disable_adapter", action="store_true", default=True)

    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    # output
    p.add_argument("--output_dir", type=str, default="/finance_ML/fengninghui/LatentMAS/outputs/lora_dapo_multiagent")
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--debug_dump_generations", action="store_true", default=True)
    p.add_argument("--debug_max_chars", type=int, default=2000)

    args = p.parse_args()

    if args.task != "medqa":
        raise ValueError("当前脚本只实现 medqa 的 0/1 reward。")

    # align HF-only flags for prompts/model wrapper
    args.method = "latent_mas"
    args.text_mas_context_length = -1
    args.device2 = args.device
    args.use_vllm = False
    args.use_second_HF_model = False
    args.enable_prefix_caching = False
    if args.latent_only:
        args.sequential_info_only = True

    set_seed(args.seed)
    device = torch.device(args.device)

    agent_roles = _parse_csv_roles(args.agent_roles)
    if not agent_roles or agent_roles[-1] != "judger":
        raise ValueError("--agent_roles must be non-empty and end with 'judger'")
    train_lora_roles = _parse_csv_roles(args.train_lora_roles)
    if not train_lora_roles:
        raise ValueError("--train_lora_roles is empty")
    bad = [r for r in train_lora_roles if r not in agent_roles]
    if bad:
        raise ValueError(f"--train_lora_roles contains roles not in --agent_roles: {bad}")

    # load model
    wrapper = ModelWrapper(args.model_name, device, use_vllm=False, args=args)
    device = torch.device(wrapper.device)

    # attach LoRA adapters (multi-adapter)
    try:
        from peft import LoraConfig, TaskType, get_peft_model  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("缺少 peft 依赖，请先 `pip install peft`") from e

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[m.strip() for m in (args.lora_target_modules or "").split(",") if m.strip()],
        bias="none",
    )
    if not hasattr(wrapper.model, "peft_config"):
        wrapper.model = get_peft_model(wrapper.model, lora_cfg)
    _ensure_adapters(wrapper.model, train_lora_roles, lora_cfg)
    _freeze_non_lora_and_select_trainable_adapters(wrapper.model, train_lora_roles)
    wrapper.model.train()

    if args.require_disable_adapter and not hasattr(wrapper.model, "disable_adapter"):
        raise RuntimeError("peft 不支持 disable_adapter()，无法严格保证 Judger 不受 LoRA 影响。")

    optimizer = torch.optim.AdamW(
        (p for p in wrapper.model.parameters() if p.requires_grad),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "train_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    # dataset
    data = _load_medqa_items_from_file(args.data_file)
    random.shuffle(data)
    if int(args.max_train_samples) > 0:
        data = data[: int(args.max_train_samples)]
    if not data:
        raise RuntimeError("Empty training data.")

    pbar = tqdm(total=int(args.train_steps), disable=bool(args.no_tqdm))
    step_time_ema: Optional[float] = None
    ema_beta = 0.9
    global_step = 0
    stats_history: List[TrainStats] = []

    def rollout_past_kv(question: str) -> Optional[Tuple]:
        past_kv: Optional[Tuple] = None
        for role in agent_roles:
            if role == "judger":
                break
            msgs = _build_messages(role, question, prompt_style=args.prompt, args=args)
            prompt = wrapper.render_chat(msgs, add_generation_prompt=True)
            if args.think:
                prompt = f"{prompt}<think>"
            enc = wrapper.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            prev_len = _past_length(past_kv)
            if role in train_lora_roles:
                wrapper.model.set_adapter(role)
                past_kv = wrapper.generate_latent_batch_grad(
                    ids, attention_mask=mask, latent_steps=int(args.latent_steps), past_key_values=past_kv
                )
            else:
                with wrapper.model.disable_adapter():
                    past_kv = wrapper.generate_latent_batch_grad(
                        ids, attention_mask=mask, latent_steps=int(args.latent_steps), past_key_values=past_kv
                    )
            if args.sequential_info_only or args.latent_only:
                new_len = _past_length(past_kv)
                tokens_added = int(new_len - prev_len)
                tokens_to_keep = int(args.latent_steps) if args.latent_only else tokens_added
                past_kv = _truncate_past_kv(past_kv, tokens_to_keep)
        return past_kv

    for step in range(1, int(args.train_steps) + 1):
        t0 = time.time()
        item = data[(step - 1) % len(data)]
        q = item["question"]
        gold = item.get("gold", "")

        skipped = False
        skip_reason = ""
        loss_val = 0.0
        avg_reward = 0.0
        reward_std = 0.0
        avg_gen_len = 0.0
        samples_used = 0

        # 1) rollout under current policy (differentiable)
        past_kv = rollout_past_kv(q)
        if past_kv is None and int(args.latent_steps) > 0:
            skipped = True
            skip_reason = "past_kv_none"
        else:
            # 2) judger prompt
            jmsgs = _build_messages("judger", q, prompt_style=args.prompt, args=args)
            jprompt = wrapper.render_chat(jmsgs, add_generation_prompt=True)
            if args.think:
                jprompt = f"{jprompt}<think>"
            jenc = wrapper.tokenizer(jprompt, return_tensors="pt", add_special_tokens=False)
            judger_ids = jenc["input_ids"].to(device)
            judger_mask = jenc["attention_mask"].to(device)
            prompt_len = int(judger_mask[0].sum().item())
            prompt_ids_trim = judger_ids[0, :prompt_len].detach().to("cpu")

            # 3) dynamic sampling group
            K0 = int(args.group_size)
            Kmax = max(K0, int(args.max_group_size))
            completions: List[torch.Tensor] = []
            texts: List[str] = []
            texts_raw: List[str] = []
            rewards: List[float] = []
            parsed: List[Optional[str]] = []
            old_logprob_norm: List[float] = []

            def has_signal(rs: List[float], preds: List[Optional[str]]) -> bool:
                if not rs:
                    return False
                r = torch.tensor(rs, dtype=torch.float32)
                if float(r.std().item()) <= float(args.min_reward_std):
                    return False
                uniq = set([p for p in preds if p is not None])
                if len(uniq) < int(args.min_unique_preds):
                    # allow reward variance even if preds same (rare), but usually this helps
                    return False
                return True

            with wrapper.model.disable_adapter():
                while len(completions) < Kmax:
                    past_kv_sampling = wrapper.detach_past_key_values(past_kv)
                    if args.per_sample_seed:
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
                    gen_text_raw = wrapper.tokenizer.decode(gen_ids, skip_special_tokens=False)
                    pred = normalize_answer(extract_mcq_choice(gen_text))
                    r = 1.0 if (pred and gold and pred == gold) else 0.0

                    completions.append(gen_ids)
                    texts.append(gen_text)
                    texts_raw.append(gen_text_raw)
                    rewards.append(r)
                    parsed.append(pred)

                    # compute old logprob under current policy snapshot (no grad), using detached cache values
                    old_lp = _compute_old_logprob_norm(
                        wrapper=wrapper,
                        past_kv_detached=past_kv_sampling,
                        prompt_ids_trim=prompt_ids_trim,
                        prompt_len=prompt_len,
                        completion_ids=gen_ids,
                        device=device,
                    )
                    old_logprob_norm.append(old_lp)

                    if not bool(args.dynamic_sampling):
                        if len(completions) >= K0:
                            break
                    else:
                        if len(completions) >= K0 and has_signal(rewards, parsed):
                            break

            samples_used = len(completions)
            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
            avg_reward = float(rewards_t.mean().item()) if samples_used else 0.0
            reward_std = float(rewards_t.std().item()) if samples_used else 0.0
            avg_gen_len = float(sum(int(c.numel()) for c in completions) / max(1, samples_used))

            if samples_used < K0:
                skipped = True
                skip_reason = "too_few_samples"
            else:
                # advantage
                if bool(args.adv_norm):
                    std = rewards_t.std().clamp_min(1e-6)
                    adv = (rewards_t - rewards_t.mean()) / std
                else:
                    adv = rewards_t - rewards_t.mean()

                if float(adv.abs().sum().item()) <= 1e-12:
                    skipped = True
                    skip_reason = "no_signal_adv"

        if skipped:
            step_time = time.time() - t0
            if step_time_ema is None:
                step_time_ema = step_time
            else:
                step_time_ema = ema_beta * step_time_ema + (1 - ema_beta) * step_time
            eta_sec = float(step_time_ema * max(0, int(args.train_steps) - step)) if step_time_ema else 0.0
            st = TrainStats(
                step=step,
                skipped=True,
                skip_reason=skip_reason,
                loss=0.0,
                avg_reward=avg_reward,
                reward_std=reward_std,
                avg_gen_len=avg_gen_len,
                samples_used=samples_used,
                epochs=int(args.dapo_epochs),
                step_time_sec=float(step_time),
                eta_sec=float(eta_sec),
            )
            stats_history.append(st)
            if not args.no_tqdm:
                finish_at = datetime.now() + timedelta(seconds=float(st.eta_sec))
                pbar.set_description(f"step {step}/{int(args.train_steps)}")
                pbar.set_postfix(
                    skip=st.skip_reason,
                    r=f"{st.avg_reward:.3f}",
                    std=f"{st.reward_std:.3f}",
                    k=str(st.samples_used),
                    t=f"{st.step_time_sec:.1f}s",
                    eta=_fmt_eta(st.eta_sec),
                    end=finish_at.strftime("%H:%M:%S"),
                )
                pbar.update(1)
            (pbar.write if not args.no_tqdm else print)(json.dumps(asdict(st), ensure_ascii=False))
            continue

        # ===== DAPO updates =====
        # we recompute rollout each epoch to preserve gradient path into role adapters
        K = samples_used
        old_lp_t = torch.tensor(old_logprob_norm, dtype=torch.float32, device=device)
        adv_t = adv.detach()  # treat advantage as constant

        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        epochs = max(1, int(args.dapo_epochs))
        for ep in range(epochs):
            # recompute differentiable past_kv under current params
            past_kv_ep = rollout_past_kv(q)
            if past_kv_ep is None and int(args.latent_steps) > 0:
                continue
            # judger base logprob with gradient through past_kv_ep
            with wrapper.model.disable_adapter():
                per_sample_losses = []
                for k in range(K):
                    full_ids = torch.cat([prompt_ids_trim, completions[k].to("cpu")], dim=0)
                    input_ids = full_ids.unsqueeze(0).to(device)
                    attn = torch.ones_like(input_ids, device=device)
                    prompt_lengths = torch.tensor([prompt_len], dtype=torch.long, device=device)
                    seq_logprobs, token_counts = wrapper.compute_completion_logprobs(
                        input_ids=input_ids,
                        attention_mask=attn,
                        prompt_lengths=prompt_lengths,
                        past_key_values=past_kv_ep,
                    )
                    new_lp = (seq_logprobs / token_counts.to(seq_logprobs.dtype)).squeeze(0)  # scalar tensor
                    ratio = torch.exp(new_lp - old_lp_t[k])
                    clipped = _decoupled_clip_ratio(ratio, adv_t[k], args.clip_eps_pos, args.clip_eps_neg)
                    # PPO-style clipped objective
                    obj1 = ratio * adv_t[k]
                    obj2 = clipped * adv_t[k]
                    obj = torch.minimum(obj1, obj2)
                    per_sample_losses.append(-obj)
                loss = torch.stack(per_sample_losses).mean()
            (loss / max(1, int(args.grad_accum_steps))).backward()
            total_loss += float(loss.detach().item())
            global_step += 1
            if global_step % max(1, int(args.grad_accum_steps)) == 0:
                if args.grad_clip and float(args.grad_clip) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        (p for p in wrapper.model.parameters() if p.requires_grad),
                        max_norm=float(args.grad_clip),
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        loss_val = float(total_loss / max(1, epochs))

        # logging
        step_time = time.time() - t0
        if step_time_ema is None:
            step_time_ema = step_time
        else:
            step_time_ema = ema_beta * step_time_ema + (1 - ema_beta) * step_time
        eta_sec = float(step_time_ema * max(0, int(args.train_steps) - step)) if step_time_ema else 0.0
        st = TrainStats(
            step=step,
            skipped=False,
            skip_reason="",
            loss=float(loss_val),
            avg_reward=float(avg_reward),
            reward_std=float(reward_std),
            avg_gen_len=float(avg_gen_len),
            samples_used=int(samples_used),
            epochs=int(epochs),
            step_time_sec=float(step_time),
            eta_sec=float(eta_sec),
        )
        stats_history.append(st)

        if not args.no_tqdm:
            finish_at = datetime.now() + timedelta(seconds=float(st.eta_sec))
            pbar.set_description(f"step {step}/{int(args.train_steps)}")
            pbar.set_postfix(
                loss=f"{st.loss:.4f}",
                r=f"{st.avg_reward:.3f}",
                std=f"{st.reward_std:.3f}",
                k=str(st.samples_used),
                ep=str(st.epochs),
                t=f"{st.step_time_sec:.1f}s",
                eta=_fmt_eta(st.eta_sec),
                end=finish_at.strftime("%H:%M:%S"),
            )
            pbar.update(1)

        (pbar.write if not args.no_tqdm else print)(json.dumps(asdict(st), ensure_ascii=False))
        preds_dbg = parsed
        (pbar.write if not args.no_tqdm else print)(f"[sample] gold={gold} rewards={rewards} preds={preds_dbg}")

        if args.debug_dump_generations:
            try:
                dbg_path = os.path.join(args.output_dir, "debug_generations.jsonl")
                keep = int(args.debug_max_chars)
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
                                    "text_clean": texts[i][-keep:] if texts[i] else "",
                                    "text_raw": texts_raw[i][-keep:] if texts_raw[i] else "",
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
            _save_role_adapters(wrapper.model, save_root=save_dir, roles=train_lora_roles)
            wrapper.tokenizer.save_pretrained(save_dir)
            with open(os.path.join(save_dir, "train_stats_tail.jsonl"), "w", encoding="utf-8") as f:
                for row in stats_history[-50:]:
                    f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")

    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    _save_role_adapters(wrapper.model, save_root=final_dir, roles=train_lora_roles)
    wrapper.tokenizer.save_pretrained(final_dir)
    with open(os.path.join(final_dir, "train_stats.jsonl"), "w", encoding="utf-8") as f:
        for row in stats_history:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")

    if not args.no_tqdm:
        pbar.close()


if __name__ == "__main__":
    main()


