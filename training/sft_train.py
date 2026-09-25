"""LoRA/QLoRA supervised fine-tuning of a small instruct model on 311 ticket data.

Loads a base instruct model (default: Qwen2.5-0.5B-Instruct), wraps it with a
LoRA adapter, and trains it to turn messy resident complaints into structured
JSON dispatch tickets. Uses 4-bit QLoRA automatically when a CUDA GPU is
available (the free-tier Colab target); falls back to full precision
otherwise, which is only meant for quick correctness smoke tests on CPU.

Peak GPU memory and wall-clock training time are recorded to
<output-dir>/run_metrics.json so the numbers can be reported to Elena.

Usage:
    python training/sft_train.py \
        --train data/train.jsonl --val data/val.jsonl \
        --output-dir checkpoints/sft-adapter
"""

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
]


def to_text(example, tokenizer):
    messages = [
        {"role": "system", "content": example["instruction"]},
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": example["output"]},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}


def build_model_and_tokenizer(model_name, use_4bit):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
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

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--val", type=Path, default=Path("data/val.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/sft-adapter"))
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
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
    model, tokenizer = build_model_and_tokenizer(args.model, use_4bit)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )

    train_ds = load_dataset("json", data_files=str(args.train), split="train")
    val_ds = load_dataset("json", data_files=str(args.val), split="train")
    train_ds = train_ds.map(to_text, fn_kwargs={"tokenizer": tokenizer})
    val_ds = val_ds.map(to_text, fn_kwargs={"tokenizer": tokenizer})

    sft_config = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        dataset_text_field="text",
        max_length=512,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_config,
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
    print(f"Training complete in {elapsed:.1f}s, peak GPU memory: {peak_mem_gb} GB")


if __name__ == "__main__":
    main()
