#!/usr/bin/env python3
"""
CPU-based LoRA fine-tuning for Qwen models.

This script trains a small LoRA adapter on CPU using PEFT.
It's designed to work with limited resources and complete within hours.
"""

import os
import json
import argparse
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType


def load_training_data(data_path: str) -> Dataset:
    """Load training data from JSONL file."""
    examples = []
    with open(data_path, 'r') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    # Convert to instruction format
    formatted = []
    for ex in examples:
        text = f"### Instruction:\n{ex['instruction']}\n\n"
        if ex.get('input'):
            text += f"### Input:\n{ex['input']}\n\n"
        text += f"### Response:\n{ex['output']}"
        formatted.append({"text": text})

    return Dataset.from_list(formatted)


def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize examples for training."""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )


def main():
    parser = argparse.ArgumentParser(description="Train LoRA adapter on CPU")
    parser.add_argument("--data", required=True, help="Path to training JSONL")
    parser.add_argument("--output", required=True, help="Output directory for adapter")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B",
                        help="Base model (use smaller for CPU)")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")

    args = parser.parse_args()

    print(f"Loading training data from {args.data}...")
    dataset = load_training_data(args.data)
    print(f"Loaded {len(dataset)} examples")

    # Use a smaller model for CPU training
    # The full 14B model is too large for CPU training
    model_name = args.base_model
    print(f"Loading base model: {model_name}")
    print("Note: Using smaller model for CPU training. Adapter can be applied to larger model.")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model in float32 for CPU
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )

    # Configure LoRA
    print(f"Configuring LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Tokenize dataset
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=dataset.column_names,
    )

    # Training arguments optimized for CPU
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=False,  # CPU doesn't support fp16
        bf16=False,
        dataloader_num_workers=0,  # Avoid multiprocessing issues
        remove_unused_columns=False,
        report_to="none",  # Disable wandb/tensorboard
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Train
    print("Starting training...")
    print(f"This will take a while on CPU. Check logs for progress.")
    trainer.train()

    # Save adapter
    print(f"Saving adapter to {args.output}...")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    # Save config for reference
    config = {
        "base_model": model_name,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "training_examples": len(dataset),
        "epochs": args.epochs,
    }
    with open(os.path.join(args.output, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("Training complete!")


if __name__ == "__main__":
    main()
