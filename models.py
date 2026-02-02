"""
`models.py` 负责把两类推理后端封装成统一接口：

1) HuggingFace Transformers 后端（`AutoModelForCausalLM`）：
   - 既能做标准的 text generation（token 空间）
   - 也能做 LatentMAS 需要的 latent rollout（隐空间：用 `inputs_embeds` 喂入隐向量）

2) vLLM 后端（`vllm.LLM`）：
   - 用于更快的最终文本生成
   - 在 LatentMAS(vLLM) 路径下，我们会把前面 agent 累积的 latent embedding 插入到 prompt embedding 里
     （见 `methods/latent_mas.py` 的 prompt_embeds 拼接逻辑）

因此 `ModelWrapper` 的核心职责是：
- 统一 tokenizer / prompt 构造
- 统一 batch 化输入
- 提供：
  - generate_text_batch（HF）
  - vllm_generate_text_batch（vLLM）
  - generate_latent_batch / generate_latent_batch_hidden_state（LatentMAS 的隐空间“通信/记忆”核心）
"""

import os
import csv
import torch
from typing import Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    # vLLM 是可选依赖：未安装时 `_HAS_VLLM=False`，项目仍可用 HF backend 跑通。
    from vllm import LLM, SamplingParams
    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False


def _ensure_pad_token(tokenizer: AutoTokenizer) -> None:
    # 很多基座模型（尤其是 chat 模型）默认没有 pad_token；
    # 但我们在 batch padding/生成时需要 pad_token_id，否则会报错/行为不稳定。
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            # 常见做法：用 EOS 作为 PAD
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})


def _past_length(past_key_values: Optional[Tuple]) -> int:
    # `past_key_values`（KV cache）结构依模型/transformers 版本不同略有差异；
    # 这里用最通用的 legacy tuple 结构取第 0 层的 K，读取其 seq_len 维度。
    if not past_key_values:
        return 0
    k = past_key_values[0][0]
    return k.shape[-2]


class ModelWrapper:
    def __init__(self, model_name: str, device: torch.device, use_vllm: bool = False, args = None):
        """
        统一封装 HF / vLLM 两条路径。

        - `model_name`：HF repo id 或本地路径（例如你使用的 `/finance_ML/.../Qwen38btext`）
        - `device`：HF 主模型所在 device（常用 `cuda:0`）
        - `use_vllm`：是否启用 vLLM（需要 pip 安装 vllm）
        - `args`：来自 run.py 的 argparse 参数（包含 vLLM/HF 双模型配置等）
        """
        self.model_name = model_name
        self.device = device
        # 只有同时满足：命令行指定 use_vllm 且环境安装了 vllm，才会走 vLLM 路径
        self.use_vllm = use_vllm and _HAS_VLLM
        self.vllm_engine = None
        # latent_space_realign 是论文里的一个对齐 trick：
        # 把 last_hidden 映射到更接近输入 embedding 的空间，提升 latent rollout 稳定性。
        self.latent_space_realign = bool(getattr(args, "latent_space_realign", False)) if args else False
        # 缓存 realign 矩阵（按 model 实例 id 做 key）
        self._latent_realign_matrices: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.args = args

        # for ablation
        self.pre_aligned = None

        if self.use_vllm:
            # ==========================
            # DEBUG BREAKPOINT SUGGESTION
            # 5) 在这里打断点：确认 vLLM 初始化参数（tp/gpu_util/max_num_seqs/max_model_len/enforce_eager）
            #    这些参数直接影响显存占用与是否 OOM。
            # ==========================
            
            tp_size = max(1, int(getattr(args, "tensor_parallel_size", 1)))
            gpu_util = float(getattr(args, "gpu_memory_utilization", 0.9))
            max_num_seqs = int(getattr(args, "vllm_max_num_seqs", 16))
            max_model_len = int(getattr(args, "vllm_max_model_len", 4096))
            enforce_eager = bool(getattr(args, "vllm_enforce_eager", False))
            
            print(f"[vLLM] Using vLLM backend for model {model_name}")
            if args.enable_prefix_caching and args.method == "latent_mas": 
                # LatentMAS(vLLM) 需要 enable_prompt_embeds=True：
                # 因为最终 judger 生成阶段，我们不是传 prompt text，而是传 prompt embeddings（含 latent memory 插入）。
                self.vllm_engine = LLM(
                    model=model_name,
                    tensor_parallel_size=tp_size,
                    gpu_memory_utilization=gpu_util,
                    max_num_seqs=max_num_seqs,
                    max_model_len=max_model_len,
                    enforce_eager=enforce_eager,
                    enable_prefix_caching=True,
                    enable_prompt_embeds=True,
                )
            else:
                self.vllm_engine = LLM(
                    model=model_name,
                    tensor_parallel_size=tp_size,
                    gpu_memory_utilization=gpu_util,
                    max_num_seqs=max_num_seqs,
                    max_model_len=max_model_len,
                    enforce_eager=enforce_eager,
                )
            # tokenizer 仍然用 HF 的 tokenizer（vLLM 也会内部初始化 tokenizer，但这里统一用 transformers 侧）
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            
            use_second_hf = bool(getattr(args, "use_second_HF_model", False)) if args else False
            if use_second_hf:
                # ==========================
                # DEBUG BREAKPOINT SUGGESTION
                # 6) 在这里打断点：确认 HF 辅助模型被放在 args.device2（通常 cuda:1）
                #    并检查 self.embedding_layer 的形状/设备。
                # ==========================
                self.HF_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    # 通常用 bf16 更省显存、更快（A100/部分卡支持 bf16）
                    torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
                ).to(args.device2).eval()
                # 用 HF 模型的 input embedding layer，把 token id 映射到 embedding（用于 judger prompt embedding 构造）
                self.embedding_layer = self.HF_model.get_input_embeddings()
                # HF 辅助模型所在 device（LatentMAS(vLLM) 通常设为 cuda:1）
                self.HF_device = args.device2
                # realign 矩阵按 HF_model 的 embedding/输出层权重构建；并缓存起来
                self._ensure_latent_realign_matrix(self.HF_model, torch.device(self.HF_device), args)
            elif self.latent_space_realign:
                raise ValueError("latent_space_realign requires --use_second_HF_model when using vLLM backend.")
            _ensure_pad_token(self.tokenizer)
            # vLLM 路径下：不加载 self.model（transformers 模型），因为最终文本生成走 vLLM；
            # latent rollout 走 self.HF_model（第二个 HF 模型）。
            return  # skip loading transformers model

        # fallback: normal transformers path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        _ensure_pad_token(self.tokenizer)
        with torch.no_grad():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
            )
        # 如果 tokenizer 新增了特殊 token（如 pad），需要 resize embedding
        if len(self.tokenizer) != self.model.get_input_embeddings().weight.shape[0]:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(device)
        self.model.eval()
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True
        if self.latent_space_realign:
            self._ensure_latent_realign_matrix(self.model, self.device, args)

    def render_chat(self, messages: List[Dict], add_generation_prompt: bool = True) -> str:
        # 将 messages（OpenAI 风格：[{role, content}, ...]）渲染成模型可接受的 prompt 文本
        # 优先使用 tokenizer.chat_template（Qwen 系列通常提供）
        tpl = getattr(self.tokenizer, "chat_template", None)
        if tpl:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        # 兜底格式：自定义一个非常简化的 chat markup
        segments = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            segments.append(f"<|{role}|>\n{content}\n</|{role}|>")
        if add_generation_prompt:
            segments.append("<|assistant|>")
        return "\n".join(segments)

    def prepare_chat_input(
        self, messages: List[Dict], add_generation_prompt: bool = True
    ) -> Tuple[str, torch.Tensor, torch.Tensor, List[str]]:
        # 单条对话 -> prompt 文本 -> token ids（含 attention_mask）
        prompt_text = self.render_chat(messages, add_generation_prompt=add_generation_prompt)
        encoded = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        # 仅把有效 token 转成人可读 token，便于 debug 打印
        active_ids = input_ids[0][attention_mask[0].bool()].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(active_ids)
        return prompt_text, input_ids, attention_mask, tokens

    def prepare_chat_batch(
        self,
        batch_messages: List[List[Dict]],
        add_generation_prompt: bool = True,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, List[List[str]]]:
        # batch 版：多条 messages -> 多条 prompt -> padding 后的 input_ids/attention_mask
        prompts: List[str] = []
        for messages in batch_messages:
            prompts.append(self.render_chat(messages, add_generation_prompt=add_generation_prompt))
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        tokens_batch: List[List[str]] = []
        for ids_row, mask_row in zip(input_ids, attention_mask):
            active_ids = ids_row[mask_row.bool()].tolist()
            tokens_batch.append(self.tokenizer.convert_ids_to_tokens(active_ids))
        return prompts, input_ids, attention_mask, tokens_batch

    def vllm_generate_text_batch(
        self,
        prompts: List[str],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> List[str]:
        # vLLM 的“纯文本 prompt -> 文本 generation”接口（baseline/text_mas 会用）
        if not self.vllm_engine:
            raise RuntimeError("vLLM engine not initialized. Pass use_vllm=True to ModelWrapper.")
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        outputs = self.vllm_engine.generate(prompts, sampling_params)
        generations = [out.outputs[0].text.strip() for out in outputs]
        return generations
    
    def _build_latent_realign_matrix(self, model, device, args) -> Tuple[torch.Tensor, torch.Tensor]:
        # Latent realignment 的核心：求一个线性映射矩阵 M，使得：
        #   hidden @ M 约等于 input_embedding 空间（并做范数归一）
        # 这里用输出 embedding（lm_head 权重）来构造一个最小二乘解。
        input_embeds = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
        output_embeds = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
        if output_embeds is None:
            output_embeds = getattr(model, "lm_head", None)
        if (
            input_embeds is None
            or output_embeds is None
            or not hasattr(input_embeds, "weight")
            or not hasattr(output_embeds, "weight")
        ):
            raise RuntimeError("Cannot build latent realignment matrix: embedding weights not accessible.")
        input_weight = input_embeds.weight.detach().to(device=device, dtype=torch.float32)
        output_weight = output_embeds.weight.detach().to(device=device, dtype=torch.float32)
        # gram = W_out^T W_out
        gram = torch.matmul(output_weight.T, output_weight)
        # 稳定数值：加一个很小的 L2 正则
        reg = 1e-5 * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        gram = gram + reg
        # rhs = W_out^T W_in
        rhs = torch.matmul(output_weight.T, input_weight)
        # 解线性方程：gram * M = rhs
        realign_matrix = torch.linalg.solve(gram, rhs)
        # 用输入 embedding 的平均范数作为目标范数（后面用于归一化）
        target_norm = input_weight.norm(dim=1).mean().detach()

        if self.args.latent_space_realign:
            pass
        else:
            # 如果不开启 realign，则退化成单位矩阵（即不做线性映射）
            realign_matrix = torch.eye(realign_matrix.shape[0], device=realign_matrix.device, dtype=realign_matrix.dtype)

        return realign_matrix, target_norm

    def _ensure_latent_realign_matrix(self, model, device, args) -> Tuple[torch.Tensor, torch.Tensor]:
        key = id(model)
        info = self._latent_realign_matrices.get(key)
        target_device = torch.device(device)

        if info is None:
            matrix, target_norm = self._build_latent_realign_matrix(model, target_device, args)
        else:
            matrix, target_norm = info
            if matrix.device != target_device:
                matrix = matrix.to(target_device)

        target_norm = target_norm.to(device=target_device, dtype=matrix.dtype) if isinstance(target_norm, torch.Tensor) else torch.as_tensor(target_norm, device=target_device, dtype=matrix.dtype)
        self._latent_realign_matrices[key] = (matrix, target_norm)

        return matrix, target_norm

    def _apply_latent_realignment(self, hidden: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        # 输入：
        # - hidden: [B, D]，通常是最后一层 hidden state 的最后一个 token
        # 输出：
        # - aligned: [B, D]，映射/归一化后的 latent 向量，可作为下一步 `inputs_embeds`
        matrix, target_norm = self._ensure_latent_realign_matrix(model, hidden.device, self.args)
        hidden_fp32 = hidden.to(torch.float32)
        aligned = torch.matmul(hidden_fp32, matrix)

        aligned_norm = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        pre_aligned = aligned.detach().clone()
        self.pre_aligned = pre_aligned
        # 把 aligned 的范数拉回 target_norm，避免 latent rollout 过程中向量爆炸/塌缩
        aligned = aligned * (target_norm / aligned_norm)
        return aligned.to(hidden.dtype)

    @torch.no_grad()
    def generate_text_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple[List[str], Optional[Tuple]]:
        # HF backend 的标准文本生成。
        # 关键点：支持传入 `past_key_values`，用于 LatentMAS 在 latent rollout 后继续解码。
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        prompt_lengths = attention_mask.sum(dim=1).tolist()
        cache_position = None
        if past_key_values is not None:
            # 当提供 past_key_values 时，需要把 cache_position 对齐到当前追加 token 的位置
            past_len = _past_length(past_key_values)
            cache_position = torch.arange(
                past_len,
                past_len + input_ids.shape[-1],
                dtype=torch.long,
                device=self.device,
            )
            if past_len > 0:
                # attention_mask 也需要补上 past 部分（全 1）
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )
        sequences = outputs.sequences
        generations: List[str] = []
        for idx, length in enumerate(prompt_lengths):
            length = int(length)
            generated_ids = sequences[idx, length:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            generations.append(text)
        return generations, outputs.past_key_values

    def tokenize_text(self, text: str) -> torch.Tensor:
        return self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(self.device)

    @torch.no_grad()
    # ==========================
    # 逐行解释（对应函数签名每一行）：
    #
    # - `@torch.no_grad()`：
    #   这一路径只做推理/评测，不需要反向传播；关闭梯度可显著减少显存占用与计算开销，
    #   并避免在 latent rollout 多步循环中累积巨大的 autograd graph。
    #
    # - `def generate_latent_batch(`：
    #   这是 LatentMAS 的“隐式推理/隐空间 rollout”入口：不生成文本 token，
    #   而是把隐藏层向量当作下一步输入 embedding，滚动 `latent_steps` 次，
    #   以此把“思考痕迹/协作信息”写入并扩展 KV cache（`past_key_values`）。
    #
    # - `self,`：
    #   访问封装好的模型/分词器/realign 矩阵缓存等成员；并可在 vLLM 模式下复用 HF_model。
    #
    # - `input_ids: torch.Tensor,`：
    #   形状通常为 [B, L]（batch, seq_len）的 token id。
    #   这是本轮 agent 的“显式提示词”（prompt）在 token 空间的表示，用来把当前问题/角色信息编码进模型状态。
    #
    # - `attention_mask: Optional[torch.Tensor] = None,`：
    #   形状通常为 [B, L]，1 表示有效 token、0 表示 padding。
    #   batch 推理时必须用它告诉模型哪些 token 是 padding；如果不传，我们会默认全 1（等价于未 padding）。
    #
    # - `*,`：
    #   Python 的“仅关键字参数”分隔符：强制后面的参数必须用 `name=value` 传入。
    #   目的：避免把 `latent_steps`/`past_key_values` 的位置传参写错（这两个参数语义很关键，写错会导致逻辑悄悄跑偏）。
    #
    # - `latent_steps: int,`：
    #   LatentMAS 的核心超参：要做多少步 latent rollout。
    #   每一步都会：
    #   1) 取当前最后位置的最后层 hidden_state（[B, D]）
    #   2) （可选）realign 到更像 embedding 的空间
    #   3) 作为 `inputs_embeds` 喂回模型（相当于追加一个“隐式 token”）
    #   从而把信息写进 KV cache，并给后续 agent / judger 使用。
    #
    # - `past_key_values: Optional[Tuple] = None,`：
    #   这就是“隐空间通信/工作记忆”的载体（KV cache）。
    #   - None：表示这是第一位 agent，从空记忆开始。
    #   - 非 None：表示已有前序 agent 的 latent rollout 结果；本函数会在其基础上继续追加 latent steps，
    #     让多个 agent 的隐式推理在同一条 KV cache 上累计（隐空间协作）。
    #
    # - `) -> Tuple:`：
    #   返回值目前主要是更新后的 `past_key_values`（有些版本注解写 Tuple 以兼容 transformers 的 cache 类型差异）。
    # ==========================
    def generate_latent_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        latent_steps: int,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple:
        # ==========================
        # LatentMAS(HF backend) 的“隐空间 rollout”
        #
        # 输入：
        # - input_ids/attention_mask：当前 agent prompt（token 空间）
        # - past_key_values：前序 agent 累积的 KV cache（代表“隐空间记忆”）
        # - latent_steps：要滚动多少步 latent（每步会把 [B,D] hidden 当成下一 token 的 embedding）
        #
        # 输出：
        # - past：更新后的 KV cache（给下一个 agent 继续用）
        # ==========================
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        else:
            attention_mask = attention_mask.to(self.device)

        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

        # 先用 token ids 走一遍 forward，拿到 hidden_states 与 KV cache
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values

        # hidden_states[0]：embedding 输出；hidden_states[-1]：最后一层输出
        # 这里只取最后一个位置（通常是 prompt 的最后 token）
        e_t = outputs.hidden_states[0][:, -1, :]          # [B, D] input embedding at last position
        last_hidden = outputs.hidden_states[-1][:, -1, :] # [B, D] last layer hidden at last position
        h_t = last_hidden.detach().clone()

        e_t_plus_1 = None
        latent_vecs_all: List[torch.Tensor] = []
        latent_vecs_all.append(e_t.detach().clone())

        for step in range(latent_steps):

            # source_model 决定 realign 矩阵用哪个模型的权重（有 HF_model 就优先）
            source_model = self.HF_model if hasattr(self, "HF_model") else self.model
            latent_vec = self._apply_latent_realignment(last_hidden, source_model)

            latent_vecs_all.append(latent_vec.detach().clone())

            if step == 0:
                e_t_plus_1 = latent_vec.detach().clone()
            
            # 把 [B, D] 扩成 [B, 1, D]，作为“下一 token”的 inputs_embeds
            latent_embed = latent_vec.unsqueeze(1)

            past_len = _past_length(past)
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=self.device,
            )
            # 注意：这里用 inputs_embeds 而非 input_ids
            # 相当于把“隐向量”当作下一步输入 token 的 embedding，继续扩展 KV cache。
            outputs = self.model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]

        return past
    
    @torch.no_grad()
    def generate_latent_batch_hidden_state(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        latent_steps: int,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple:
        # ==========================
        # LatentMAS(vLLM backend) 辅助函数：
        # - 仍然用 HF_model 做 latent rollout（因为 vLLM 不支持修改 KV/直接 latent rollout）
        # - 同时把“每一步喂进去的 embedding 序列”收集起来，供 vLLM 侧做 prompt_embeds 插入。
        #
        # 输出：
        # - past：HF_model 的 KV cache
        # - concatenated embeddings：shape [B, 1 + latent_steps, D]
        #   第 0 个是 input embedding（对应原 prompt 的最后位置），后面每个是 latent step 的 embedding。
        # ==========================
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.HF_device)
        else:
            attention_mask = attention_mask.to(self.HF_device)
        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
        outputs = self.HF_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        
        curr_output_embedding = [] 
        # 记录：input embedding（这里是整段 prompt 的 embedding 序列，后续会 cat）
        curr_output_embedding.append(outputs.hidden_states[0])  # [B, L_prompt, D]
        
        
        for _ in range(latent_steps):

            source_model = self.HF_model if hasattr(self, "HF_model") else self.model
            latent_vec = self._apply_latent_realignment(last_hidden, source_model)
            latent_embed = latent_vec.unsqueeze(1)
            past_len = _past_length(past)
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=latent_embed.device,
            )
            outputs = self.HF_model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]

            # 记录每一步 latent 输入 embedding（[B, 1, D]）
            curr_output_embedding.append(latent_embed.detach())

        # 拼接得到 [B, L_prompt + latent_steps, D]（注意：这里把整段 prompt embedding 也拼进去了）
        return past, torch.cat(curr_output_embedding, dim=1) # Output input embeddings

