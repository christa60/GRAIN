"""
Fine-tuning Llama-3.2-3B-Instruct on GSM8K (Grade School Math)
=========================================================
Uses QLoRA (4-bit quantization + LoRA adapters) via HuggingFace + TRL.
Chat template updated for Llama-3.2-Instruct format:
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>

    {system}<|eot_id|>
    <|start_header_id|>user<|end_header_id|>

    {question}<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>

    {answer}<|eot_id|>

Requirements:
    pip install torch transformers datasets trl peft bitsandbytes accelerate wandb

Recommended hardware: 1x A100/H100 (40GB+) or 2x A6000 for Llama-3.2-3B.

Usage:
    # Basic run
    python train_gsm8k_llama32.py

    # With custom args
    python train_gsm8k_llama32.py \
        --model_name meta-llama/Llama-3.2-3B-Instruct \
        --output_dir ./llama32-gsm8k \
        --num_train_epochs 3 \
        --per_device_train_batch_size 4 \
        --use_wandb True

    accelerate launch train_gsm8k_llama32.py \
        --model_name meta-llama/Llama-3.2-3B-Instruct \
        --output_dir ./llama32-gsm8k/acc/42 \
        --num_train_epochs 6 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 2 \
        --use_wandb False \
        --seed 42

    accelerate launch finetune_gsm8k_llama32.py \
        --model_name meta-llama/Llama-3.1-8B \
        --output_dir ./llama31-8B-gsm8k/acc/42 \
        --num_train_epochs 6 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 2 \
        --use_wandb False \
        --seed 42
"""

import argparse
import os
import re
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
)
from trl import SFTConfig, SFTTrainer
from minnorm_trainer import MinNormSFTTrainer
from huggingface_hub import login

login(token="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")


# ---------------------------------------------------------------------------
# Prompt formatting  ←  LLAMA-3.2-INSTRUCT TEMPLATE
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful math tutor. Solve the following grade school math problem "
    "step by step. At the end of your solution, state the final answer as:\n"
    "#### <number>"
)

# Special tokens used by Llama-3.2-Instruct
BOS             = "<|begin_of_text|>"
EOT             = "<|eot_id|>"
START_HEADER    = "<|start_header_id|>"
END_HEADER      = "<|end_header_id|>"

# Sentinel used to split prompt from answer during inference
ASSISTANT_HEADER = f"{START_HEADER}assistant{END_HEADER}\n\n"


def _header(role: str) -> str:
    """Return the role header block, e.g. <|start_header_id|>user<|end_header_id|>\n\n"""
    return f"{START_HEADER}{role}{END_HEADER}\n\n"


def format_example(example: dict) -> str:
    """
    Convert a GSM8K example into the Llama-3.2-Instruct prompt+completion string.

    Format (single-turn with system prompt):
        <|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>

        {system}<|eot_id|>
        <|start_header_id|>user<|end_header_id|>

        {question}<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>

        {answer}<|eot_id|>

    The full string (including the final <|eot_id|>) is used as the training
    target so the model learns when to stop generating.
    """
    question = example["question"].strip()
    answer   = example["answer"].strip()   # GSM8K: CoT reasoning + "#### N"

    return (
        f"{BOS}"
        f"{_header('system')}{SYSTEM_PROMPT}{EOT}"
        f"{_header('user')}{question}{EOT}"
        f"{ASSISTANT_HEADER}{answer}{EOT}"
    )


def make_inference_prompt(text: str) -> str:
    """
    Strip the answer portion from a formatted training string so the model
    can complete it during evaluation.

    Training string ends with:  ...assistant\\n\\n{answer}<|eot_id|>
    Inference string ends with: ...assistant\\n\\n
    (generation starts cleanly right after the assistant header)
    """
    # Split on the assistant header and keep everything up to and including it
    parts = text.split(ASSISTANT_HEADER)
    if len(parts) < 2:
        # Fallback: return as-is (should not happen with well-formed data)
        return text
    return parts[0] + ASSISTANT_HEADER


def extract_final_answer(text: str) -> Optional[str]:
    """Pull the number after '####' from a model or reference answer."""
    match = re.search(r"####\s*([\d,\.]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    return None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ScriptArgs:
    model_name: str = field(
        default="meta-llama/Llama-3.2-3B-Instruct",
        metadata={"help": "HuggingFace model id or local path"},
    )
    output_dir: str = field(default="./llama32-gsm8k-qlora")
    dataset_name: str = field(default="openai/gsm8k")
    dataset_config: str = field(default="main")

    # LoRA
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated list of modules to apply LoRA to"},
    )

    # Quantization
    load_in_4bit: bool = field(default=True)
    bnb_4bit_compute_dtype: str = field(default="bfloat16")
    bnb_4bit_quant_type: str = field(default="nf4")
    use_double_quant: bool = field(default=True)

    # Training
    num_train_epochs: int = field(default=3)
    per_device_train_batch_size: int = field(default=4)
    per_device_eval_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=4)
    learning_rate: float = field(default=2e-4)
    warmup_ratio: float = field(default=0.03)
    lr_scheduler_type: str = field(default="cosine")
    max_seq_length: int = field(default=1024)
    weight_decay: float = field(default=0.01)
    max_grad_norm: float = field(default=1.0)

    # Eval / logging
    eval_steps: int = field(default=100)
    logging_steps: int = field(default=25)
    save_steps: int = field(default=200)
    save_total_limit: int = field(default=3)

    # Misc
    seed: int = field(default=42)
    use_wandb: bool = field(default=False)
    wandb_project: str = field(default="llama32-gsm8k")
    push_to_hub: bool = field(default=False)
    hub_model_id: Optional[str] = field(default=None)


# ---------------------------------------------------------------------------
# Build model + tokenizer
# ---------------------------------------------------------------------------

def build_model_and_tokenizer(args: ScriptArgs):
    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.load_in_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=args.use_double_quant,
    )

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config if args.load_in_4bit else None,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
        device_map={"": local_rank},
    )

    model.config.use_cache = False
    # Note: pretraining_tp is Llama-2 specific, not needed for Llama-3.2
    # model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )

    # Llama-3.2 has no pad token by default — use EOS as pad
    tokenizer.pad_token = tokenizer.eos_token
    # Right-padding is required for causal LM training with bf16
    tokenizer.padding_side = "right"

    return model, tokenizer


# ---------------------------------------------------------------------------
# Build LoRA config
# ---------------------------------------------------------------------------

def build_lora_config(args: ScriptArgs) -> LoraConfig:
    target_modules = [m.strip() for m in args.lora_target_modules.split(",")]
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def build_datasets(args: ScriptArgs, tokenizer):
    raw = load_dataset(args.dataset_name, args.dataset_config)
    train_ds = raw["train"]
    test_ds  = raw["test"]

    def preprocess(example):
        return {"text": format_example(example)}

    train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names)
    test_ds  = test_ds.map(preprocess,  remove_columns=test_ds.column_names)

    print(f"Train size: {len(train_ds):,}  |  Test size: {len(test_ds):,}")
    print("\nSample formatted prompt:\n")
    print(train_ds[0]["text"])
    print("\n" + "-" * 60)

    return train_ds, test_ds


# ---------------------------------------------------------------------------
# Evaluation helper (exact-match on final numeric answer)
# ---------------------------------------------------------------------------

def evaluate_gsm8k(
    model,
    tokenizer,
    dataset,
    num_samples: int = 200,
    batch_size: int = 8,
) -> dict:
    """
    Run greedy decoding on `num_samples` test examples and compute
    exact-match accuracy on the final numeric answer.
    """
    model.eval()
    # Left-padding at inference so all generated tokens are right-aligned
    tokenizer.padding_side = "left"

    correct = 0
    total   = 0

    samples = dataset.select(range(min(num_samples, len(dataset))))

    for i in tqdm(range(0, len(samples), batch_size), desc="Evaluating"):
        batch = samples.select(range(i, min(i + batch_size, len(samples))))

        # Build inference prompts (question only, no answer)
        prompts      = [make_inference_prompt(ex["text"]) for ex in batch]
        gold_answers = [
            extract_final_answer(ex["text"].split(ASSISTANT_HEADER, 1)[-1])
            for ex in batch
        ]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        for j, output in enumerate(outputs):
            decoded = tokenizer.decode(output, skip_special_tokens=True)
            pred    = extract_final_answer(decoded)
            if pred is not None and pred == gold_answers[j]:
                correct += 1
            total += 1

    # Restore right-padding for any subsequent training
    tokenizer.padding_side = "right"

    accuracy = correct / total if total > 0 else 0.0
    print(f"\nGSM8K Eval — Accuracy: {correct}/{total} = {accuracy:.2%}")
    return {"exact_match": accuracy, "correct": correct, "total": total}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    for f in ScriptArgs.__dataclass_fields__.values():
        parser.add_argument(
            f"--{f.name}",
            type=(
                type(f.default)
                if not isinstance(f.default, bool)
                else lambda x: x.lower() == "true"
            ),
            default=f.default,
            help=f.metadata.get("help", ""),
        )
    cli_args = parser.parse_args()
    args     = ScriptArgs(**vars(cli_args))

    set_seed(args.seed)

    # W&B
    if args.use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args))
    else:
        os.environ["WANDB_DISABLED"] = "true"

    # Model + tokenizer
    model, tokenizer = build_model_and_tokenizer(args)

    # LoRA
    lora_config = build_lora_config(args)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Data
    train_ds, test_ds = build_datasets(args, tokenizer)

    # Training arguments
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        report_to="wandb" if args.use_wandb else "none",
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        dataset_text_field="text",
        max_length=args.max_seq_length,
        packing=True,   # pack multiple short examples into one sequence for efficiency
        ddp_find_unused_parameters=False,
    )

    trainer = MinNormSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        processing_class=tokenizer,
    )

    per_device = trainer.args.per_device_train_batch_size
    grad_acc   = trainer.args.gradient_accumulation_steps
    world_size = trainer.args.world_size
    total_batch = per_device * grad_acc * world_size
    print(f"TOTAL BATCH SIZE: {total_batch}")

    # Train
    print("\n=== Starting Training ===")
    trainer.train()

    # Save final adapter
    final_adapter_path = os.path.join(args.output_dir, "final_adapter")
    trainer.model.save_pretrained(final_adapter_path)
    tokenizer.save_pretrained(final_adapter_path)
    print(f"\nAdapter saved to: {final_adapter_path}")

    # Final eval
    print("\n=== Running Final GSM8K Evaluation ===")
    results = evaluate_gsm8k(
        trainer.model,
        tokenizer,
        test_ds,
        num_samples=200,
        batch_size=args.per_device_eval_batch_size,
    )

    if args.use_wandb:
        import wandb
        wandb.log({"final_gsm8k_exact_match": results["exact_match"]})
        wandb.finish()

    return results


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Expected formatted sample (Llama-3.2-Instruct template):
# ---------------------------------------------------------------------------
# <|begin_of_text|><|start_header_id|>system<|end_header_id|>
#
# You are a helpful math tutor. Solve the following grade school math problem
# step by step. At the end of your solution, state the final answer as:
# #### <number><|eot_id|>
# <|start_header_id|>user<|end_header_id|>
#
# Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning
# and bakes muffins for her friends every day with four. She sells the remainder
# at the farmers' market daily for $2 per fresh duck egg. How much in dollars
# does she make every day at the farmers' market?<|eot_id|>
# <|start_header_id|>assistant<|end_header_id|>
#
# She has 16 - 3 - 4 = 9 eggs left. She makes 9 * 2 = $18 per day. #### 18<|eot_id|>