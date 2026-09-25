"""Short DPO pass on top of the SFT LoRA adapter.

Continues training the SAME LoRA adapter produced by sft_train.py (rather
than adding a second adapter on top), teaching it to prefer well-formed
dispatch tickets over plausible-but-flawed ones (wrong priority, confused
category, vague/verbatim details, chatty non-JSON wrapping, etc.).

Passing ref_model=None lets TRL use the peft "disable adapter" trick to get
reference log-probs from the frozen base model, so no second full copy of
the model needs to be held in memory for the reference policy.

Peak GPU memory and wall-clock time are recorded to
<output-dir>/run_metrics.json.

Usage:
    python training/dpo_train.py \
        --sft-adapter checkpoints/sft-adapter \
        --train data/dpo_train.jsonl --val data/dpo_val.jsonl \
        --output-dir checkpoints/dpo-adapter
"""

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer


def build_model_and_tokenizer(model_name, adapter_dir, use_4bit):
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    dtype = torch.float32
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        dtype = torch.bfloat16

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=True)
    return model, tokenizer


def to_prompt_completion(example):
    return {
        "prompt": [
            {"role": "system", "content": example["instruction"]},
            {"role": "user", "content": example["input"]},
        ],
        "chosen": [{"role": "assistant", "content": example["chosen"]}],
        "rejected": [{"role": "assistant", "content": example["rejected"]}],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--sft-adapter", type=Path, default=Path("checkpoints/sft-adapter"))
    parser.add_argument("--train", type=Path, default=Path("data/dpo_train.jsonl"))
    parser.add_argument("--val", type=Path, default=Path("data/dpo_val.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/dpo-adapter"))
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument(
        "--no-4bit", action="store_true",
        help="disable QLoRA 4-bit loading (use for CPU debugging/smoke tests)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=-1,
        help="cap training steps, for quick smoke tests (-1 = full run)",
    )
    args = parser.parse_args()

    use_4bit = torch.cuda.is_available() and not args.no_4bit
    model, tokenizer = build_model_and_tokenizer(args.model, str(args.sft_adapter), use_4bit)

    train_ds = load_dataset("json", data_files=str(args.train), split="train")
    val_ds = load_dataset("json", data_files=str(args.val), split="train")
    train_ds = train_ds.map(to_prompt_completion, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(to_prompt_completion, remove_columns=val_ds.column_names)

    dpo_config = DPOConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        max_length=512,
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None

    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    metrics = {"wall_clock_seconds": elapsed, "peak_gpu_memory_gb": peak_mem_gb}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "run_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"DPO training complete in {elapsed:.1f}s, peak GPU memory: {peak_mem_gb} GB")


if __name__ == "__main__":
    main()
