from typing import Dict, List, Optional, Tuple

from . import default_agents
from models import ModelWrapper, _past_length
from prompts import build_agent_message_sequential_latent_mas, build_agent_message_hierarchical_latent_mas
from utils import extract_gsm8k_answer, normalize_answer, extract_markdown_python_block, run_with_timeout, extract_mcq_choice
import torch
import argparse
import pdb

try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = None

try:
    from vllm import SamplingParams  # type: ignore
    _HAS_VLLM = True
except ImportError:
    SamplingParams = None  # type: ignore
    _HAS_VLLM = False

class LatentMASMethod:
    def __init__(
        self,
        model: ModelWrapper,
        *,
        latent_steps: int = 10,
        judger_max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        generate_bs: int = 1,
        args: argparse.Namespace = None,
    ) -> None:
        self.args = args
        self.model = model
        self.latent_steps = latent_steps
        self.judger_max_new_tokens = judger_max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.generate_bs = max(1, generate_bs)
        self.agents = default_agents()
        self.method_name = 'latent_mas'
        self.vllm_device = args.device 
        self.HF_device = args.device2
        self.latent_only = bool(getattr(args, "latent_only", False)) if args else False
        self.sequential_info_only = bool(getattr(args, "sequential_info_only", False)) if args else False

        if self.latent_only:
            self.sequential_info_only = True

        if SamplingParams is None:
            self.sampling_params = None
        else:
            self.sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=args.max_new_tokens,
            )
        self.task = args.task

    @staticmethod
    def _slice_tensor(tensor: torch.Tensor, tokens_to_keep: int) -> torch.Tensor:
        if tokens_to_keep <= 0:
            return tensor[..., 0:0, :].contiguous()
        keep = min(tokens_to_keep, tensor.shape[-2])
        start = tensor.shape[-2] - keep
        return tensor[..., start:, :].contiguous()

    def _truncate_past(self, past_kv: Optional[Tuple], tokens_to_keep: int) -> Optional[Tuple]:
        if past_kv is None or tokens_to_keep <= 0:
            return None
        if Cache is not None and isinstance(past_kv, Cache):
            legacy = past_kv.to_legacy_cache()
            trimmed_legacy = tuple(
                tuple(self._slice_tensor(t, tokens_to_keep) for t in layer)
                for layer in legacy
            )
            return past_kv.__class__.from_legacy_cache(trimmed_legacy)
        trimmed_layers = []
        for layer in past_kv:
            if isinstance(layer, tuple):
                trimmed_layers.append(tuple(self._slice_tensor(t, tokens_to_keep) for t in layer))
            elif torch.is_tensor(layer):
                trimmed_layers.append(self._slice_tensor(layer, tokens_to_keep))
            else:
                trimmed_layers.append(layer)
        return tuple(trimmed_layers)

    @torch.no_grad()
    def run_batch(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        past_kv: Optional[Tuple] = None
        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]

        for agent in self.agents:

            if self.args.prompt == "sequential":
                batch_messages = [
                    build_agent_message_sequential_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args)
                    for item in items
                ]
            elif self.args.prompt == "hierarchical":
                batch_messages = [
                    build_agent_message_hierarchical_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args)
                    for item in items
                ]


            # ==========================
            # 下面这段是 LatentMAS(HF backend) 的核心：多 Agent 的“隐空间通信”如何实现。
            #
            # 总体思路：
            # - 对每个 agent 都构造一份“同一个问题 + 不同角色指令”的 prompt
            # - 非 Judger agent 不做文本解码，而是做 latent rollout：
            #   通过 `ModelWrapper.generate_latent_batch(..., past_key_values=past_kv)` 把隐式推理写入 KV cache
            # - `past_kv` 会在 agent 之间传递并累积 => 相当于共享一段“隐空间工作记忆”
            # - 最后 Judger 带着累积好的 `past_kv` 调 `generate_text_batch` 产出最终自然语言答案
            #
            # 你不能 debug 时，可以用注释理解两个最关键对象：
            # - past_kv：KV cache（隐空间记忆/通信载体）
            # - wrapped_prompts：实际喂给模型的 prompt 文本（可能会追加 <think>）
            # ==========================

            # 把一批 message（OpenAI 风格 role/content）渲染为 prompt 文本，并 tokenization 成批张量
            # 返回：
            # - prompts: List[str]               每条样本的 prompt 文本
            # - input_ids: torch.Tensor [B, L]   padding 后的 token ids
            # - attention_mask: [B, L]           1=有效 token, 0=padding
            # - tokens_batch: List[List[str]]    仅用于打印/debug（把有效 token id 转成 token 字符串）
            prompts, input_ids, attention_mask, tokens_batch = self.model.prepare_chat_batch(
                batch_messages, add_generation_prompt=True
            )

            # 非 Judger agent：只做 latent rollout（隐式推理），不生成可见文本。
            if agent.role != "judger":
                # 记录进入本 agent 前的 KV cache 长度（用于后面判断“本 agent 新增了多少 token/latent”）
                prev_past_len = _past_length(past_kv)

                # 可选开关：在 prompt 末尾追加 "<think>"。
                # 目的：对一些模型（尤其 Qwen chat 模型），“显式进入思考模式”可能改变 hidden states 的分布，
                # 从而影响 latent rollout 的质量/稳定性。
                if self.args.think:
                    wrapped_prompts = [f"{prompt}<think>" for prompt in prompts]
                else:
                    wrapped_prompts = prompts

                # 注意：这里“重新 tokenize” wrapped_prompts，而不是复用上面的 input_ids/attention_mask。
                # 原因：上面 prepare_chat_batch 得到的 prompts 还没加 "<think>"；
                # 一旦加了 "<think>"，token ids / mask 都会变化，必须重新编码才能一致。
                wrapped_encoded = self.model.tokenizer(
                    wrapped_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                # 这里走 HF backend，所以张量放到 self.model.device（通常 cuda:0）
                wrapped_ids = wrapped_encoded["input_ids"].to(self.model.device)
                wrapped_mask = wrapped_encoded["attention_mask"].to(self.model.device)

                # 把有效 token id 转成 token（仅用于保存 trace 方便观察，不参与计算）
                wrapped_tokens_batch: List[List[str]] = []
                for ids_row, mask_row in zip(wrapped_ids, wrapped_mask):
                    active_ids = ids_row[mask_row.bool()].tolist()
                    wrapped_tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))

                # 关键调用：隐空间 rollout（隐式推理）
                #
                # 语义：
                # - 先用 wrapped_ids / wrapped_mask 把“本 agent 的 prompt”写入模型状态
                # - 再循环 latent_steps 次：
                #   取最后层最后位置 hidden_state（[B, D]）->（可选 realign）-> 作为 inputs_embeds 喂回模型
                #   这样 KV cache 会被持续“追加”，形成一段只存在于隐空间的协作轨迹
                #
                # 最重要的就是 `past_key_values=past_kv`：
                # - None 表示从空记忆开始
                # - 非 None 表示在前序 agent 的“隐空间记忆”基础上继续追加
                past_kv = self.model.generate_latent_batch(
                    wrapped_ids,
                    attention_mask=wrapped_mask,
                    latent_steps=self.latent_steps,
                    past_key_values=past_kv,
                )

                # 可选的 KV 裁剪策略：
                # - sequential_info_only：仅保留“本 agent 新增的信息”（避免 KV 过长导致显存/速度问题）
                # - latent_only：更激进，只保留 latent_steps 对应的最后几步“隐式 token”
                if self.sequential_info_only or self.latent_only:
                    # new_past_len：本次 latent rollout 后 KV cache 的总长度
                    new_past_len = _past_length(past_kv)
                    # tokens_added：本 agent 贡献的 KV 增量（包含 prompt token + latent_steps）
                    tokens_added = new_past_len - prev_past_len
                    # tokens_to_keep：
                    # - latent_only: 只保留最后 latent_steps（相当于丢掉显式 prompt token 的影响）
                    # - sequential_info_only: 保留本 agent 增量 tokens_added（相当于丢掉更早 agent 的历史）
                    tokens_to_keep = self.latent_steps if self.latent_only else tokens_added
                    # 实际裁剪 KV cache（对每层 K/V 做切片）
                    past_kv = self._truncate_past(past_kv, tokens_to_keep)

                # 记录 trace：把本 agent 的输入（prompt+可选 think）和 latent_steps 记下来
                # 这里 output 为空字符串，是因为非 Judger agent 不输出文本（“隐式推理”）
                for idx in range(batch_size):
                    mask = wrapped_mask[idx].bool()
                    trimmed_ids = wrapped_ids[idx][mask].to("cpu").tolist()
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": wrapped_prompts[idx],
                            "input_ids": trimmed_ids,
                            "input_tokens": wrapped_tokens_batch[idx],
                            "latent_steps": self.latent_steps,
                            "output": "",
                        }
                    )
            else:
                # Judger agent：负责把隐空间记忆“解码”为可读的最终文本答案。
                #
                # past_for_decoding：
                # - latent_steps > 0：说明前面 agent 产生了隐空间记忆，需要把它带入解码阶段
                # - latent_steps == 0：退化为无 latent 的普通推理，past_kv 没意义，传 None 即可
                past_for_decoding = past_kv if self.latent_steps > 0 else None

                # 与前面一致：可选给 Judger 的 prompt 追加 <think>
                if self.args.think:
                    judger_prompts = [f"{prompt}<think>" for prompt in prompts]
                else:
                    judger_prompts = prompts
                
                # 对 Judger 的 prompt 做编码（同样需要重新 tokenize，因为可能加了 <think>）
                judger_encoded = self.model.tokenizer(
                    judger_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                judger_ids = judger_encoded["input_ids"].to(self.model.device)
                judger_mask = judger_encoded["attention_mask"].to(self.model.device)

                # 把 Judger prompt 的有效 token 转成 token（仅用于 trace 保存）
                judger_tokens_batch: List[List[str]] = []
                for ids_row, mask_row in zip(judger_ids, judger_mask):
                    active_ids = ids_row[mask_row.bool()].tolist()
                    judger_tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))

                # 关键：最终解码
                # - 输入仍是 Judger 的显式 prompt（token 空间）
                # - 但同时带上 past_key_values=past_for_decoding
                #   => 模型会在已有 KV cache（隐空间记忆）基础上继续生成输出 token
                generated_batch, _ = self.model.generate_text_batch(
                    judger_ids,
                    judger_mask,
                    max_new_tokens=self.judger_max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    past_key_values=past_for_decoding,
                )
                for idx in range(batch_size):
                    final_text = generated_batch[idx].strip()
                    final_texts[idx] = final_text
                    mask = judger_mask[idx].bool()
                    trimmed_ids = judger_ids[idx][mask].to("cpu").tolist()
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": judger_prompts[idx],
                            "input_ids": trimmed_ids,
                            "input_tokens": judger_tokens_batch[idx],
                            "output": final_text,
                        }
                    )

        results: List[Dict] = []
        for idx, item in enumerate(items):
            final_text = final_texts[idx]
            if self.task in ['mbppplus', 'humanevalplus']:
                pred = extract_markdown_python_block(final_text)
                gold = item.get("gold", "")

                if pred is None:
                    ok = False
                    error_msg = "python error: No python code block found"
                else:
                    python_code_to_exe = pred + "\n" + gold
                    ok, error_msg = run_with_timeout(python_code_to_exe, timeout=10)
                
                print(f'=========================================')
                print(f'Question {idx}')
                print(f'error_msg: {error_msg}')
                # print(f'=========================================')

            elif self.task in ["aime2024", "aime2025"]:
                pred = normalize_answer(extract_gsm8k_answer(final_text))
                gold = str(item.get("gold", "")).strip()
                try:
                    pred_int = int(pred)
                    gold_int = int(gold)
                    ok = (pred_int == gold_int)
                    error_msg = None
                except ValueError:
                    ok = False
                    error_msg = f'Value error in parsing answer. Pred: {pred}, Gold: {gold}'

            else:
                if self.task in ["medqa", "arc_easy", "arc_challenge"]:
                    pred = normalize_answer(extract_mcq_choice(final_text))
                else:
                    pred = normalize_answer(extract_gsm8k_answer(final_text))
                gold = item.get("gold", "")
                ok = (pred == gold) if (pred and gold) else False
                error_msg = None
            
            results.append(
                {
                    "question": item["question"],
                    "gold": gold,
                    "solution": item["solution"],
                    "prediction": pred,
                    "raw_prediction": final_text,
                    "agents": agent_traces[idx],
                    "correct": ok,
                }
            )
        return results
    
    def run_batch_vllm(self, items: List[Dict]) -> List[Dict]:
        if not _HAS_VLLM or self.sampling_params is None or getattr(self.model, "vllm_engine", None) is None:
            raise RuntimeError(
                "vLLM backend is not available. Please `pip install vllm` and run with `--use_vllm`."
            )
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        past_kv: Optional[Tuple] = None
        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]

        embedding_record = []
        for agent in self.agents:
            
            if self.args.prompt == "sequential":
                batch_messages = [
                    build_agent_message_sequential_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args)
                    for item in items
                ]
            elif self.args.prompt == "hierarchical":
                batch_messages = [
                    build_agent_message_hierarchical_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args)
                    for item in items
                ]
                
            prompts, input_ids, attention_mask, tokens_batch = self.model.prepare_chat_batch(
                batch_messages, add_generation_prompt=True
            )

            if agent.role != "judger":
                prev_past_len = _past_length(past_kv)

                # to wrap all latent thoughts from previous agents
                if self.args.think:
                        wrapped_prompts = [f"{prompt}<think>" for prompt in prompts]
                else: 
                    wrapped_prompts = prompts

                wrapped_encoded = self.model.tokenizer(
                    wrapped_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                wrapped_ids = wrapped_encoded["input_ids"].to(self.model.HF_device)
                wrapped_mask = wrapped_encoded["attention_mask"].to(self.model.HF_device)
                wrapped_tokens_batch: List[List[str]] = []
                for ids_row, mask_row in zip(wrapped_ids, wrapped_mask):
                    active_ids = ids_row[mask_row.bool()].tolist()
                    wrapped_tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))

                past_kv, previous_hidden_embedding = self.model.generate_latent_batch_hidden_state(
                    wrapped_ids,
                    attention_mask=wrapped_mask,
                    latent_steps=self.latent_steps,
                    past_key_values=past_kv,
                )
                # ==========================
                # DEBUG BREAKPOINT SUGGESTION
                # 7) 在这里打断点：观察“隐空间通信”的核心产物
                #    - previous_hidden_embedding: shape [B, L_latent(+1), H]（包含 input embedding + latent steps）
                #    - past_kv: HF 模型的 KV cache
                # ==========================
                if self.sequential_info_only or self.latent_only:
                    new_past_len = _past_length(past_kv)
                    tokens_added = new_past_len - prev_past_len
                    tokens_to_keep = self.latent_steps if self.latent_only else tokens_added
                    past_kv = self._truncate_past(past_kv, tokens_to_keep)

                if self.latent_only:
                    if self.latent_steps > 0:
                        previous_hidden_embedding = previous_hidden_embedding[:, -self.latent_steps:, :]
                    else:
                        previous_hidden_embedding = previous_hidden_embedding[:, 0:0, :]

                embedding_record.append(previous_hidden_embedding)

                if self.sequential_info_only or self.latent_only:
                    embedding_record = embedding_record[-1:]
                
                for idx in range(batch_size):
                    mask = wrapped_mask[idx].bool()
                    trimmed_ids = wrapped_ids[idx][mask].to("cpu").tolist()
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": wrapped_prompts[idx],
                            "input_ids": trimmed_ids,
                            "input_tokens": wrapped_tokens_batch[idx],
                            "latent_steps": self.latent_steps,
                            "output": "",
                        }
                    )
            else:
                
                # A stack of [B, L_i, H]
                past_embedding = torch.cat(embedding_record, dim=1).to(self.vllm_device)
                # ==========================
                # DEBUG BREAKPOINT SUGGESTION
                # 8) 在这里打断点：past_embedding 是所有非 Judger agent 的 latent 记忆拼接结果
                #    shape 预期 [B, sum(L_i), H]，设备应为 vllm_device（通常 cuda:0）
                # ==========================
                
                if self.args.think:
                    judger_prompts = [f"{prompt}<think>" for prompt in prompts]
                else: 
                    judger_prompts = prompts
                
                judger_encoded = self.model.tokenizer(
                    judger_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                ) 
                judger_encoded = judger_encoded["input_ids"].to(self.model.HF_device)
                # Get current prompt embedding
                # Keep batch dimension (B, L, H). Do NOT squeeze, otherwise B is dropped when B==1.
                curr_prompt_emb = self.model.embedding_layer(judger_encoded).to(self.vllm_device)
                # ==========================
                # DEBUG BREAKPOINT SUGGESTION
                # 9) 在这里打断点：检查 curr_prompt_emb / past_embedding 的 H 是否一致、dtype 是否一致
                # ==========================
                
                # assert Qwen model
                assert "Qwen" in self.args.model_name or "qwen" in self.args.model_name, "latent_embedding_position is only supported for Qwen models currently."

                # handle latent embedding insertion position    
                len_of_left = []
                for p in judger_prompts:
                    idx = p.find("<|im_start|>user\n")
                    # Get the text up to and including "<|im_start|>user\n"
                    left = p[: idx + len("<|im_start|>user\n")]
                    len_of_left.append(len(self.model.tokenizer(left)['input_ids']))
                    
                B, L, H = curr_prompt_emb.shape
                _, Lp, H = past_embedding.shape  # assume shape consistency
                    
                whole_prompt_emb_list = []
                for i in range(B):
                    insert_idx = len_of_left[i]
                    left_emb = curr_prompt_emb[i, :insert_idx, :]
                    right_emb = curr_prompt_emb[i, insert_idx:, :]
                    combined = torch.cat([left_emb, past_embedding[i], right_emb], dim=0)
                    whole_prompt_emb_list.append(combined)
                # ==========================
                # DEBUG BREAKPOINT SUGGESTION
                # 10) 在这里打断点：确认 latent embedding 被插入的位置 insert_idx 是否合理
                #     （代码目前基于 "<|im_start|>user\\n" 来定位插入点，仅对 Qwen 模型模板适配）
                # ==========================

                # Pad back to max length if needed
                max_len = max(x.shape[0] for x in whole_prompt_emb_list)
                whole_prompt_emb = torch.stack([
                    torch.cat([x, torch.zeros(max_len - x.shape[0], H, device=x.device)], dim=0)
                    for x in whole_prompt_emb_list
                ])

                # else:
                    # Get full prompt embedding from cat with previous ones 
                    # B L H B L H
                    # whole_prompt_emb = torch.cat([past_embedding, curr_prompt_emb], dim=1)
                
                # pdb.set_trace()              
                
                # Use vLLM 
                prompt_embeds_list = [
                    {
                        "prompt_embeds": embeds
                    } for embeds in whole_prompt_emb 
                ]
                
                
                outputs = self.model.vllm_engine.generate(
                    prompt_embeds_list,
                    self.sampling_params,
                )
                # ==========================
                # DEBUG BREAKPOINT SUGGESTION
                # 11) 在这里打断点：查看 vLLM 返回的 outputs/out.outputs[0].text
                # ==========================

                generated_texts = [out.outputs[0].text.strip() for out in outputs]
                    
                for idx in range(batch_size):
                    text_out = generated_texts[idx].strip()
                    final_texts[idx] = text_out
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": judger_prompts[idx],
                            "output": text_out,
                        }
                    )


        results: List[Dict] = []
        for idx, item in enumerate(items):
            final_text = final_texts[idx]
            if self.task in ["medqa", "arc_easy", "arc_challenge"]:
                pred = normalize_answer(extract_mcq_choice(final_text))
            else:
                pred = normalize_answer(extract_gsm8k_answer(final_text))
            gold = item["gold"]
            ok = (pred == gold) if (pred and gold) else False
            results.append(
                {
                    "question": item["question"],
                    "gold": gold,
                    "solution": item["solution"],
                    "prediction": pred,
                    "raw_prediction": final_text,
                    "agents": agent_traces[idx],
                    "correct": ok,
                }
            )
        return results

    def run_item(self, item: Dict) -> Dict:
        return self.run_batch([item])[0]
