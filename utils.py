import os
import random
import re
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def auto_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# this is to extract answer in \boxed{}
def extract_gsm8k_answer(text: str) -> Optional[str]:
    boxes = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxes:
        content = boxes[-1]
        number = re.search(r"[-+]?\d+(?:\.\d+)?", content)
        return number.group(0) if number else content.strip()

    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1]
    return None


def extract_gold(text: str) -> Optional[str]:
    match = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", text)
    return match.group(1) if match else None


def normalize_answer(ans: Optional[str]) -> Optional[str]:
    if ans is None:
        return None
    return ans.strip().lower()


def extract_mcq_choice(text: str) -> Optional[str]:
    """
    Extract A/B/C/D style multiple-choice answer from model output.
    Returns lowercase 'a'/'b'/'c'/'d' when found.
    Designed to avoid false-matching the leading article "A " in prompts.
    """
    if not text:
        return None

    # 1) Prefer boxed answers: \boxed{C}
    m = re.search(r"\\boxed\{\s*([A-Da-d])\s*\}", text)
    if m:
        return m.group(1).lower()

    # ==========================
    # DEBUG BREAKPOINT SUGGESTION
    # 12) 在这里打断点：如果你发现评测 pred 为 None/错判，
    #     直接查看 text 的结尾几行，确认模型到底输出了什么格式的答案。
    # ==========================

    # 2) Common explicit forms: (C), [C]
    m = re.search(r"[\(\[]\s*([A-Da-d])\s*[\)\]]", text)
    if m:
        return m.group(1).lower()

    # 3) "Option C" / "Answer: C"
    m = re.search(r"\b(?:option|answer)\s*[:=]?\s*([A-Da-d])\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower()

    # 4) Trailing single-letter answer (avoid matching leading 'A ' in the question)
    m = re.search(r"(?:^|\n)\s*([A-Da-d])\s*[\.\s]*$", text.strip())
    if m:
        return m.group(1).lower()

    return None


def extract_markdown_python_block(text: str) -> Optional[str]:
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return None


# to run python
import traceback
from multiprocessing import Process, Manager
def run_with_timeout(code, timeout):
    def worker(ns, code):
        try:
            local_ns = {}
            exec(code, local_ns)
            ns['ok'] = True
            ns['error'] = None
        except Exception:
            ns['ok'] = False
            ns['error'] = traceback.format_exc()
    with Manager() as manager:
        ns = manager.dict()
        p = Process(target=worker, args=(ns, code))
        p.start()
        p.join(timeout)
        if p.is_alive():
            p.terminate()
            ns['ok'] = False
            ns['error'] = f"TimeoutError: Execution exceeded {timeout} seconds"
        return ns.get('ok', False), ns.get('error', None)

