"""
Fine-tuning Mistral on PubMedQA
=================================
Uses QLoRA (4-bit quantization + LoRA adapters) via HuggingFace + TRL.

Requirements:
    pip install torch transformers datasets trl peft bitsandbytes accelerate wandb

Usage:
    accelerate launch finetune_pubmed_mistral.py \
        --model_name mistralai/Mistral-7B-Instruct-v0.3 \
        --output_dir ./mistral-pubmedqa/acc/42 \
        --num_train_epochs 10 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 4 \
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
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend, FineGrainedFP8Config



# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

# Mistral Instruct uses [INST] / [/INST] chat delimiters, not header tokens.
# The model was not trained with an explicit system role, so we prepend the
# system content inside the first [INST] block (common practice).

SYSTEM_PROMPT = (
    "You are a biomedical research assistant. "
    "Given a clinical question and related research context, respond with one of: "
    "'yes', 'no', or 'maybe', then provide a concise explanation grounded in the context.\n"
    "Format your response as:\n"
    "Answer: <yes|no|maybe>\n\n"
    "Explanation: <your reasoning>"
)

# Max characters per context paragraph — keeps sequences inside max_seq_length
MAX_CONTEXT_CHARS = 600


def format_example(example: dict) -> str:
    """Convert a PubMedQA example into a Mistral Instruct chat string."""
    contexts = example["context"]["contexts"]
    context_text = "\n\n".join(
        f"[{i + 1}] {c[:MAX_CONTEXT_CHARS]}" for i, c in enumerate(contexts)
    )

    user_content = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Question: {example['question'].strip()}\n\n"
        f"Research Context:\n{context_text}"
    )

    assistant_content = (
        f"Answer: {example['final_decision']}\n\n"
        f"Explanation: {example['long_answer'].strip()}"
    )

    # Mistral Instruct v1/v2/v3 format:
    # <s>[INST] {user} [/INST] {assistant}</s>
    return (
        f"<s>[INST] {user_content} [/INST] "
        f"{assistant_content}</s>"
    )


def extract_decision(text: str) -> Optional[str]:
    """Pull the yes/no/maybe decision from a model or reference answer."""
    match = re.search(r"Answer:\s*(yes|no|maybe)", text, re.IGNORECASE)
    return match.group(1).lower() if match else None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ScriptArgs:
    model_name: str = field(
        default="mistralai/Mistral-7B-Instruct-v0.3",
        metadata={"help": "HuggingFace model id or local path"},
    )
    output_dir: str = field(default="./mistral-pubmedqa-qlora")
    dataset_name: str = field(default="pubmed_qa")
    dataset_config: str = field(default="pqa_labeled")

    # LoRA — Mistral shares the same attention/MLP projection names as LLaMA
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
    per_device_train_batch_size: int = field(default=2)
    per_device_eval_batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=8)
    learning_rate: float = field(default=5e-5)
    warmup_ratio: float = field(default=0.03)
    lr_scheduler_type: str = field(default="cosine")
    max_seq_length: int = field(default=1024)
    weight_decay: float = field(default=0.01)
    max_grad_norm: float = field(default=0.3)

    # Eval / logging
    eval_steps: int = field(default=50)
    logging_steps: int = field(default=10)
    save_steps: int = field(default=100)
    save_total_limit: int = field(default=3)

    # Misc
    seed: int = field(default=42)
    use_wandb: bool = field(default=False)
    wandb_project: str = field(default="mistral-pubmedqa")
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
    model = Mistral3ForConditionalGeneration.from_pretrained(
        args.model_name,
        quantization_config=bnb_config if args.load_in_4bit else None,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
        device_map={"": local_rank},
        # attn_implementation="flash_attention_2",  # uncomment if flash-attn is installed
    )

    model.config.use_cache = False
    # pretraining_tp > 1 is only meaningful for certain Llama checkpoints; keep at 1
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=False,
    )

    # Mistral tokenizer does not always set a pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
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

    # pqa_labeled only ships a train split — carve out 20% for eval
    split    = raw["train"].train_test_split(test_size=0.2, seed=42)
    train_ds = split["train"]
    test_ds  = split["test"]

    def preprocess(example):
        return {"text": format_example(example)}

    train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names)
    test_ds  = test_ds.map(preprocess,  remove_columns=test_ds.column_names)

    print(f"Train size: {len(train_ds):,}  |  Test size: {len(test_ds):,}")
    print("\nSample prompt:\n", train_ds[0]["text"][:500], "\n...")
    return train_ds, test_ds


# ---------------------------------------------------------------------------
# Evaluation helper (exact-match on yes/no/maybe decision)
# ---------------------------------------------------------------------------

def evaluate_pubmedqa(
    model, tokenizer, dataset, num_samples: int = 100, batch_size: int = 4
) -> dict:
    """
    Run greedy decoding on `num_samples` test examples and compute
    exact-match accuracy on the final yes/no/maybe decision.
    """
    model.eval()
    tokenizer.padding_side = "left"
    correct = 0
    total = 0

    def make_inference_prompt(example: dict) -> str:
        """Strip the assistant turn so the model must generate it."""
        # Everything before [/INST] plus the delimiter itself
        text = example["text"]
        inst_end = text.find("[/INST]")
        if inst_end == -1:
            return text  # fallback
        return text[: inst_end + len("[/INST]")] + " "

    samples = dataset.select(range(min(num_samples, len(dataset))))

    for i in tqdm(range(0, len(samples), batch_size)):
        batch = samples.select(range(i, min(i + batch_size, len(samples))))
        prompts = [make_inference_prompt(ex) for ex in batch]

        # Gold labels are in the assistant portion (after [/INST])
        gold_answers = []
        for ex in batch:
            text = ex["text"]
            inst_end = text.find("[/INST]")
            assistant_text = text[inst_end + len("[/INST]"):] if inst_end != -1 else text
            gold_answers.append(extract_decision(assistant_text))

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args_global.max_seq_length,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        for j, output in enumerate(outputs):
            decoded = tokenizer.decode(output, skip_special_tokens=True)
            pred = extract_decision(decoded)
            if pred is not None and pred == gold_answers[j]:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0.0
    print(f"\nPubMedQA Eval — Accuracy: {correct}/{total} = {accuracy:.2%}")
    return {"exact_match": accuracy, "correct": correct, "total": total}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

args_global: ScriptArgs = None


def main():
    global args_global

    parser = argparse.ArgumentParser()
    for f in ScriptArgs.__dataclass_fields__.values():
        parser.add_argument(
            f"--{f.name}",
            type=type(f.default) if not isinstance(f.default, bool) else lambda x: x.lower() == "true",
            default=f.default,
            help=f.metadata.get("help", ""),
        )
    cli_args = parser.parse_args()
    args = ScriptArgs(**vars(cli_args))
    args_global = args

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
        packing=True,  # pack multiple short examples into one sequence for efficiency
        ddp_find_unused_parameters=True,
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
    print("TOTAL BATCH SIZE:", total_batch)

    # Train
    print("\n=== Starting Training ===")
    trainer.train()

    # Save final adapter
    final_adapter_path = os.path.join(args.output_dir, "final_adapter")
    trainer.model.save_pretrained(final_adapter_path)
    tokenizer.save_pretrained(final_adapter_path)
    print(f"\nAdapter saved to: {final_adapter_path}")

    # Final eval
    print("\n=== Running Final PubMedQA Evaluation ===")
    results = evaluate_pubmedqa(
        trainer.model,
        tokenizer,
        test_ds,
        num_samples=len(test_ds),
        batch_size=args.per_device_eval_batch_size,
    )

    if args.use_wandb:
        import wandb
        wandb.log({"final_pubmedqa_exact_match": results["exact_match"]})
        wandb.finish()

    return results


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Expected formatted sample (Mistral Instruct format):
# ---------------------------------------------------------------------------
# <s>[INST] You are a biomedical research assistant. Given a clinical question
# and related research context, respond with one of: 'yes', 'no', or 'maybe',
# then provide a concise explanation grounded in the context.
# Format your response as:
# Answer: <yes|no|maybe>
#
# Explanation: <your reasoning>
#
# Question: Does mitochondrial dysfunction contribute to Alzheimer's disease?
#
# Research Context:
# [1] Mitochondrial dysfunction has been observed in the brains of Alzheimer's
# patients, including reduced activity of key enzymes...
# [2] Animal models show that mitochondrial impairment precedes amyloid plaque
# formation, suggesting a causal role... [/INST]
# Answer: yes
#
# Explanation: Multiple lines of evidence support mitochondrial dysfunction as
# a contributor to Alzheimer's pathology...</s>