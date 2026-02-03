"""
Merge sharded eval outputs produced by eval_lora.py (--num_shards/--shard_id).

It reads preds_*.jsonl files and recomputes overall accuracy.

Examples:
  python merge_eval_shards.py --input_dir outputs/eval_compare --mode compare --num_shards 2
  python merge_eval_shards.py --input_dir outputs/eval_base --mode single --tag base --num_shards 8
"""

import argparse
import json
import os
from typing import Dict, List


def _read_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _accuracy(rows: List[Dict]) -> Dict[str, float]:
    total = len(rows)
    correct = sum(1 for r in rows if r.get("correct", False))
    acc = correct / total if total else 0.0
    return {"total": total, "correct": correct, "accuracy": acc}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--num_shards", type=int, required=True)
    p.add_argument("--mode", choices=["single", "compare"], required=True)
    p.add_argument("--tag", type=str, default=None, help="single 模式下：base 或 lora")
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    out_path = args.output or os.path.join(args.input_dir, "merged_summary.json")

    if args.mode == "single":
        if not args.tag:
            raise ValueError("--tag is required for mode=single")
        all_rows: List[Dict] = []
        for sid in range(int(args.num_shards)):
            fn = f"preds_{args.tag}_shard{sid}of{int(args.num_shards)}.jsonl"
            path = os.path.join(args.input_dir, fn)
            all_rows.extend(_read_jsonl(path))
        summary = {"mode": "single", "tag": args.tag, **_accuracy(all_rows)}
    else:
        base_rows: List[Dict] = []
        lora_rows: List[Dict] = []
        for sid in range(int(args.num_shards)):
            base_fn = f"preds_base_shard{sid}of{int(args.num_shards)}.jsonl"
            lora_fn = f"preds_lora_shard{sid}of{int(args.num_shards)}.jsonl"
            base_rows.extend(_read_jsonl(os.path.join(args.input_dir, base_fn)))
            lora_rows.extend(_read_jsonl(os.path.join(args.input_dir, lora_fn)))
        base_s = _accuracy(base_rows)
        lora_s = _accuracy(lora_rows)
        summary = {
            "mode": "compare",
            "base": base_s,
            "lora": lora_s,
            "delta_acc": float(lora_s["accuracy"] - base_s["accuracy"]),
        }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()


