"""
GRPO training for multi-agent LatentMAS (Planner -> Critic -> Refiner -> Judger),
with **per-role LoRA adapters** (optional) and **end-to-end differentiable latent rollout**.

Design goals (aligned with this repo's LatentMAS implementation):
- Agent roles/order follow `methods.default_agents()` by default:
  planner, critic, refiner, judger
- Non-judger agents perform *latent rollout* only (no visible text generation),
  writing implicit reasoning into `past_key_values` (KV cache).
- Judger is kept "unchanged" by disabling adapters during sampling and logprob computation.
- Gradients flow through the implicit KV cache to selected trainable role adapters
  (e.g., train planner+refiner) without being broken by discrete sampling.

Compared to `train_grpo_agent1.py`:
- This script supports full multi-agent rollout (planner/critic/refiner before judger),
  and optional LoRA adapters per role (select via `--train_lora_roles`).
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
from tqdm.auto import tqdm
from datasets import load_dataset

from methods import default_agents
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
    This mirrors `LatentMASMethod._truncate_past` but avoids any detach,
    so gradients (w.r.t. earlier trainable adapters) can still flow through the kept slice.
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
    """
    Freeze everything except LoRA params for the adapters listed in `trainable_roles`.
    This enables per-role fine-tuning within a single PeftModel with multiple adapters.
    """
    # Freeze all by default
    for _, p in model.named_parameters():
        p.requires_grad = False

    # Enable LoRA params for selected adapters
    trainable_roles = list(trainable_roles)
    for name, p in model.named_parameters():
        if "lora_" not in name:
            continue
        # PEFT parameter names usually include adapter name segments like ".planner." or ".default."
        if any(f".{role}." in name for role in trainable_roles):
            p.requires_grad = True


def _ensure_adapters(model, trainable_roles: Sequence[str], lora_cfg) -> None:
    """
    Ensure the model has adapters named by `trainable_roles`.
    We intentionally do NOT create adapters for frozen roles; they will run with adapters disabled.
    """
    existing = set(getattr(model, "peft_config", {}).keys()) if hasattr(model, "peft_config") else set()
    for role in trainable_roles:
        if role in existing:
            continue
        # add_adapter exists on PeftModel
        model.add_adapter(role, lora_cfg)


def _save_role_adapters(model, *, save_root: str, roles: Sequence[str]) -> None:
    """
    Save each role adapter into its own subdirectory:
      save_root/<role>/
    This makes evaluation/loading unambiguous for multi-adapter training.
    """
    roles = list(roles)
    if not roles:
        return
    for role in roles:
        out_dir = os.path.join(save_root, role)
        os.makedirs(out_dir, exist_ok=True)
        try:
            # Newer PEFT supports saving selected adapters explicitly.
            model.save_pretrained(out_dir, selected_adapters=[role])
        except TypeError:
            # Fallback: switch active adapter then save (older PEFT).
            try:
                model.set_adapter(role)
            except Exception:
                pass
            model.save_pretrained(out_dir)


def _load_medqa_items_from_file(data_file: str) -> List[Dict]:
    """
    Load MedQA-like JSON in the same format as `data/medqa.json`:
      - item["query"] (question text)
      - item["options"] (list of option strings)
      - item["answer"] (a substring that appears in the correct option)

    Returns rows shaped like the rest of this repo expects:
      {"question": str, "solution": str, "gold": "a"|"b"|"c"|"d"}
    """
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


def main() -> None:
    if sys.version_info >= (3, 12):
        raise RuntimeError(
            f"检测到 Python {sys.version.split()[0]}（{sys.executable}）。请使用 py310 环境运行。"
        )

    parser = argparse.ArgumentParser()

    # core
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--task", type=str, default="medqa", choices=["medqa"])
    parser.add_argument(
        "--data_file",
        type=str,
        default="/finance_ML/fengninghui/LatentMAS/data/medqa.json",
        help="训练数据集路径（MedQA json，默认使用仓库内 data/medqa.json）。",
    )
    parser.add_argument("--prompt", type=str, default="sequential", choices=["sequential", "hierarchical"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--device_map", type=str, default=None, help="多卡省显存：auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_tqdm", action="store_true")

    # agents
    parser.add_argument(
        "--agent_roles",
        type=str,
        default="planner,critic,refiner,judger",
        help="按顺序执行的 agent roles（逗号分隔）。通常保持默认以对齐原 LatentMAS。",
    )
    parser.add_argument(
        "--train_lora_roles",
        type=str,
        default="planner",
        help="哪些角色启用 LoRA 并参与微调（逗号分隔）。例：planner,refiner",
    )

    # latent rollout
    parser.add_argument("--latent_steps", type=int, default=2)
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--latent_space_realign", action="store_true")
    parser.add_argument("--sequential_info_only", action="store_true")
    parser.add_argument("--latent_only", action="store_true")

    # sampling
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--typical_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--group_size", type=int, default=2)
    parser.add_argument("--per_sample_seed", action="store_true")

    # train
    parser.add_argument("--train_steps", type=int, default=50)
    parser.add_argument("--max_train_samples", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--tf_sequential", action="store_true", default=True)

    # LoRA
    parser.add_argument("--lora_init_path", type=str, default=None, help="（可选）载入已有 LoRA（单 adapter）继续训练")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument(
        "--require_disable_adapter",
        action="store_true",
        default=True,
        help="严格模式：必须支持 disable_adapter()（确保 Judger 不受 LoRA 影响）。",
    )

    # output
    parser.add_argument("--output_dir", type=str, default="/finance_ML/fengninghui/LatentMAS/outputs/lora_grpo_multiagent")
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--debug_dump_generations", action="store_true", default=True)
    parser.add_argument("--debug_max_chars", type=int, default=2000)

    args = parser.parse_args()

    if args.task != "medqa":
        raise ValueError("当前脚本只实现 medqa 的 0/1 reward。")

    # Align flags (same convention as other scripts)
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
    if not agent_roles:
        raise ValueError("--agent_roles is empty")
    if "judger" not in agent_roles:
        raise ValueError("--agent_roles must contain 'judger' (final role) to align with LatentMAS.")
    if agent_roles[-1] != "judger":
        raise ValueError("为对齐原 LatentMAS，请把 'judger' 放在 agent_roles 的最后一位。")

    train_lora_roles = _parse_csv_roles(args.train_lora_roles)
    if not train_lora_roles:
        raise ValueError("--train_lora_roles is empty. If you want no training, don't run this script.")
    bad = [r for r in train_lora_roles if r not in agent_roles]
    if bad:
        raise ValueError(f"--train_lora_roles contains roles not in --agent_roles: {bad}")

    # 1) load model
    wrapper = ModelWrapper(args.model_name, device, use_vllm=False, args=args)
    device = torch.device(wrapper.device)

    # 2) attach LoRA (multi-adapter)
    try:
        import peft
        from peft import LoraConfig, TaskType, get_peft_model, PeftModel
    except Exception as e:  # pragma: no cover
        raise RuntimeError("缺少 peft 依赖，请先 `pip install peft`") from e

    if args.lora_init_path:
        # Note: loading a multi-adapter checkpoint is non-trivial; this path is mainly for single-adapter continuation.
        wrapper.model = PeftModel.from_pretrained(wrapper.model, args.lora_init_path, is_trainable=True)

    target_modules = [m.strip() for m in (args.lora_target_modules or "").split(",") if m.strip()]
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
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
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "train_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    data = _load_medqa_items_from_file(args.data_file)
    random.shuffle(data)
    if args.max_train_samples > 0:
        data = data[: int(args.max_train_samples)]
    if not data:
        raise RuntimeError(f"Empty training data. Check --data_file={args.data_file}")

    pad_id = int(wrapper.tokenizer.pad_token_id)
    step_time_ema: Optional[float] = None
    ema_beta = 0.9
    global_step = 0
    stats_history: List[TrainStats] = []
    pbar = tqdm(total=int(args.train_steps), disable=bool(args.no_tqdm))

    # Print a concise adapter summary for sanity
    if hasattr(wrapper.model, "peft_config"):
        adapters = list(getattr(wrapper.model, "peft_config", {}).keys())
        (pbar.write if not args.no_tqdm else print)(f"[adapters] existing={adapters} trainable={train_lora_roles}")

    for step in range(1, int(args.train_steps) + 1):
        t0 = time.time()
        item = data[(step - 1) % len(data)]
        q = item["question"]
        gold = item.get("gold", "")

        # 1) multi-agent differentiable latent rollout -> past_kv (with graph)
        past_kv: Optional[Tuple] = None
        for role in agent_roles:
            if role == "judger":
                break

            msgs = _build_messages(role, q, prompt_style=args.prompt, args=args)
            prompt = wrapper.render_chat(msgs, add_generation_prompt=True)
            if args.think:
                prompt = f"{prompt}<think>"

            enc = wrapper.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)

            prev_len = _past_length(past_kv)

            if role in train_lora_roles:
                # enable this role's adapter
                try:
                    wrapper.model.set_adapter(role)
                except Exception as e:
                    raise RuntimeError(f"Failed to set_adapter('{role}'). Existing adapters={getattr(wrapper.model,'peft_config',{}).keys()}") from e
                past_kv = wrapper.generate_latent_batch_grad(
                    ids,
                    attention_mask=mask,
                    latent_steps=int(args.latent_steps),
                    past_key_values=past_kv,
                )
            else:
                # frozen role: disable adapters, but KEEP grad enabled so gradients can flow through past_kv
                with wrapper.model.disable_adapter():
                    past_kv = wrapper.generate_latent_batch_grad(
                        ids,
                        attention_mask=mask,
                        latent_steps=int(args.latent_steps),
                        past_key_values=past_kv,
                    )

            if args.sequential_info_only or args.latent_only:
                new_len = _past_length(past_kv)
                tokens_added = int(new_len - prev_len)
                tokens_to_keep = int(args.latent_steps) if args.latent_only else tokens_added
                past_kv = _truncate_past_kv(past_kv, tokens_to_keep)

        if past_kv is None and int(args.latent_steps) > 0:
            raise RuntimeError("past_kv is None after rollout; check agent_roles/latent_steps.")

        # Important: sampling must not mutate the cache object.
        # transformers Cache may be mutated in-place during generate(); if we reuse the same cache across K samples,
        # later samples won't start from the same context and may collapse diversity.
        # Therefore we create a fresh detached cache per sample.

        # 2) judger prompt tokenization (judger always disables adapters)
        judger_msgs = _build_messages("judger", q, prompt_style=args.prompt, args=args)
        judger_prompt = wrapper.render_chat(judger_msgs, add_generation_prompt=True)
        if args.think:
            judger_prompt = f"{judger_prompt}<think>"
        jenc = wrapper.tokenizer(judger_prompt, return_tensors="pt", add_special_tokens=False)
        judger_ids = jenc["input_ids"].to(device)
        judger_mask = jenc["attention_mask"].to(device)
        prompt_len = int(judger_mask[0].sum().item())
        prompt_ids_trim = judger_ids[0, :prompt_len].detach().to("cpu")

        # 3) sample K completions with Judger(base)
        K = int(args.group_size)
        completions: List[torch.Tensor] = []
        texts: List[str] = []
        texts_raw: List[str] = []
        rewards: List[float] = []

        with wrapper.model.disable_adapter():
            for _ in range(K):
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

        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        adv = rewards_t - rewards_t.mean()

        # 4) teacher-forcing logprob with Judger(base), but past_kv keeps graph => grads to selected role adapters
        optimizer.zero_grad(set_to_none=True)
        losses = []
        with wrapper.model.disable_adapter():
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

        (loss / max(1, int(args.grad_accum_steps))).backward()
        global_step += 1
        if global_step % max(1, int(args.grad_accum_steps)) == 0:
            if args.grad_clip and float(args.grad_clip) > 0:
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

        (pbar.write if not args.no_tqdm else print)(json.dumps(asdict(st), ensure_ascii=False))
        preds_dbg = [normalize_answer(extract_mcq_choice(t)) for t in texts]
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
