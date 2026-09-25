"""Evaluate a model (optionally with a LoRA adapter) on the 311 ticket task.

Generates a structured ticket for each complaint in an eval file and scores
it against the ground-truth ticket: does it parse as valid JSON with the
right schema, and do category/priority match. Writes per-example generations
to <output>.jsonl and a summary (including generation wall-clock time) to
<output>.summary.json.

Run once per model variant to compare them:
    python training/evaluate.py --label base --output eval_results/base
    python training/evaluate.py --label sft --adapter checkpoints/sft-adapter --output eval_results/sft
    python training/evaluate.py --label dpo --adapter checkpoints/dpo-adapter --output eval_results/dpo
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REQUIRED_FIELDS = {"category", "priority", "location", "summary", "details", "requested_action"}
CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def strip_code_fence(text):
    return CODE_FENCE_RE.sub("", text.strip()).strip()


def load_model(model_name, adapter_dir):
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir or model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, device_map="auto" if torch.cuda.is_available() else None,
    )
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tokenizer


def generate_ticket(model, tokenizer, instruction, complaint, max_new_tokens):
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": complaint},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    completion_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(completion_ids, skip_special_tokens=True).strip()


def score(generated_text, gold_ticket):
    try:
        parsed = json.loads(strip_code_fence(generated_text))
    except json.JSONDecodeError:
        return {
            "valid_json": False, "has_all_fields": False,
            "category_match": False, "priority_match": False, "exact_match": False,
        }
    return {
        "valid_json": True,
        "has_all_fields": REQUIRED_FIELDS.issubset(parsed.keys()),
        "category_match": parsed.get("category") == gold_ticket.get("category"),
        "priority_match": parsed.get("priority") == gold_ticket.get("priority"),
        "exact_match": parsed == gold_ticket,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--adapter", type=Path, default=None,
        help="LoRA adapter directory; omit to evaluate the base model",
    )
    parser.add_argument("--eval-file", type=Path, default=Path("data/val.jsonl"))
    parser.add_argument("--label", default="base", help="name for this run, used in the summary")
    parser.add_argument("--output", type=Path, default=Path("eval_results/run"))
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--limit", type=int, default=None, help="only evaluate the first N examples")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, str(args.adapter) if args.adapter else None)

    examples = [json.loads(line) for line in args.eval_file.open(encoding="utf-8")]
    if args.limit:
        examples = examples[: args.limit]

    results = []
    start = time.time()
    for ex in examples:
        generated = generate_ticket(model, tokenizer, ex["instruction"], ex["input"], args.max_new_tokens)
        gold_ticket = json.loads(ex["output"])
        results.append({
            "input": ex["input"],
            "gold": ex["output"],
            "generated": generated,
            **score(generated, gold_ticket),
        })
    elapsed = time.time() - start

    n = len(results)
    summary = {
        "label": args.label,
        "n_examples": n,
        "wall_clock_seconds": elapsed,
        "seconds_per_example": elapsed / n if n else None,
        "valid_json_rate": sum(r["valid_json"] for r in results) / n,
        "has_all_fields_rate": sum(r["has_all_fields"] for r in results) / n,
        "category_accuracy": sum(r["category_match"] for r in results) / n,
        "priority_accuracy": sum(r["priority_match"] for r in results) / n,
        "exact_match_rate": sum(r["exact_match"] for r in results) / n,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with Path(f"{args.output}.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with Path(f"{args.output}.summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
