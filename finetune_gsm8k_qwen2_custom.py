"""
Fine-tuning Llama on GSM8K (Grade School Math)
================================================
Uses QLoRA (4-bit quantization + LoRA adapters) via HuggingFace + TRL.

Requirements:
    pip install torch transformers datasets trl peft bitsandbytes accelerate wandb

Recommended hardware: 1x A100/H100 (40GB+) or 2x A6000 for Llama-3-8B.
For Llama-3-70B you'll want 4x A100 80GB or use FSDP/DeepSpeed.

Usage:
    # Basic run
    python train_gsm8k.py

    # With custom args
    python train_gsm8k.py \
        --model_name meta-llama/Meta-Llama-3-8B \
        --output_dir ./llama3-gsm8k \
        --num_train_epochs 3 \
        --per_device_train_batch_size 4 \
        --use_wandb

    accelerate launch finetune_gsm8k_custom.py \
        --model_name Qwen/Qwen2-7B-Instruct \
        --output_dir ./qwen-gsm8k/acc2/42x \
        --num_train_epochs 0 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 2 \
        --use_wandb False \
        --seed 42 \
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



# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful math tutor. Solve the following grade school math problem "
    "step by step. At the end of your solution, state the final answer as:\n"
    "#### <number>"
)


def format_example(example: dict) -> str:
    """Convert a GSM8K example into a chat-style prompt+completion string."""
    question = example["question"].strip()
    answer = example["answer"].strip()  # already contains chain-of-thought + #### N

    # GSM8K answers look like:  "She has 3 apples...\n#### 3"
    # We keep the full CoT answer as the target.
    return (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n{question}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n{answer}<|eot_id|>"
    )


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
        default="meta-llama/Llama-2-7b-hf",
        metadata={"help": "HuggingFace model id or local path"},
    )
    output_dir: str = field(default="./llama2-gsm8k-qlora")
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
    max_grad_norm: float = field(default=0.3)

    # Eval / logging
    eval_steps: int = field(default=100)
    logging_steps: int = field(default=25)
    save_steps: int = field(default=200)
    save_total_limit: int = field(default=3)

    # Misc
    seed: int = field(default=42)
    use_wandb: bool = field(default=False)
    wandb_project: str = field(default="llama-gsm8k")
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

    import os
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    print(f"Loading model: {args.model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config if args.load_in_4bit else None,
        torch_dtype=compute_dtype,
        # device_map="auto",
        trust_remote_code=True,
        device_map={"": local_rank},
        # attn_implementation="flash_attention_2",  # remove if flash-attn not installed
    )

    model = PeftModel.from_pretrained(
        base_model,
        "./qwen-gsm8k/grad_acc/42/final_adapter"
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
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
    train_ds = raw["train"]
    test_ds = raw["test"]

    def preprocess(example):
        return {"text": format_example(example)}

    train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names)
    test_ds = test_ds.map(preprocess, remove_columns=test_ds.column_names)

    print(f"Train size: {len(train_ds):,}  |  Test size: {len(test_ds):,}")
    print("\nSample prompt:\n", train_ds[0]["text"][:500], "\n...")
    return train_ds, test_ds


# ---------------------------------------------------------------------------
# Evaluation helper (exact-match on final numeric answer)
# ---------------------------------------------------------------------------

import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

def evaluate_gsm8k(model, tokenizer, dataset, num_samples: int = 200, batch_size: int = 8) -> dict:
    """
    Distributed-safe eval. Each GPU processes a shard of the dataset,
    then results are gathered across ranks via all_reduce.
    """
    model.eval()
    tokenizer.padding_side = "left"

    def make_inference_prompt(text: str) -> str:
        question = text.split("<|start_header_id|>assistant<|end_header_id|>")[0]
        return question + "<|start_header_id|>assistant<|end_header_id|>\n"

    # ── shard the dataset ────────────────────────────────────────────────
    samples = dataset.select(range(min(num_samples, len(dataset))))

    is_distributed = dist.is_available() and dist.is_initialized()

    sampler = DistributedSampler(
        samples,
        num_replicas=dist.get_world_size() if is_distributed else 1,
        rank=dist.get_rank()               if is_distributed else 0,
        shuffle=False,
    ) if is_distributed else None

    loader = DataLoader(
        samples,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
    )

    correct = 0
    total   = 0

    for batch in tqdm(loader):
        # DataLoader collates {"text": [...]} — extract the list of strings
        texts = batch["text"]

        prompts = [make_inference_prompt(t) for t in texts]
        gold_answers = [
            extract_final_answer(
                t.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
            )
            for t in texts
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
                temperature=1.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        for j, output in enumerate(outputs):
            decoded = tokenizer.decode(output, skip_special_tokens=True)
            pred    = extract_final_answer(decoded)
            correct += int(pred is not None and pred == gold_answers[j])
            total   += 1

    # ── gather counts from all ranks ─────────────────────────────────────
    if is_distributed:
        correct_t = torch.tensor(correct, dtype=torch.long, device=model.device)
        total_t   = torch.tensor(total,   dtype=torch.long, device=model.device)
        dist.all_reduce(correct_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_t,   op=dist.ReduceOp.SUM)
        correct = correct_t.item()
        total   = total_t.item()

    # ── only rank 0 prints and returns meaningful results ─────────────────
    if not is_distributed or dist.get_rank() == 0:
        accuracy = correct / total if total > 0 else 0.0
        print(f"\nGSM8K Eval — Accuracy: {correct}/{total} = {accuracy:.2%}")
        return {"exact_match": accuracy, "correct": correct, "total": total}

    return {"exact_match": 0.0, "correct": 0, "total": 0}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
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
    grad_acc = trainer.args.gradient_accumulation_steps
    world_size = trainer.args.world_size
    total_batch = per_device * grad_acc * world_size
    print("TOTAL BATCH SIZE:", total_batch)
    # print(per_device, grad_acc, world_size)

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
    results = evaluate_gsm8k(trainer.model, tokenizer, test_ds, num_samples=200, batch_size=args.per_device_eval_batch_size)

    if args.use_wandb:
        import wandb
        wandb.log({"final_gsm8k_exact_match": results["exact_match"]})
        wandb.finish()

    return results


if __name__ == "__main__":
    main()

# ["<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful math tutor. Solve the following grade school math problem step by step. At the end of your solution, state the final answer as:\n#### <number><|eot_id|><|start_header_id|>user<|end_header_id|>\nJanet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"]
# {'input_ids': tensor([[   27,    91,  7265,  3575,  4326,    91,  1784,    91,  2468,  8757,
#            842,    91,    29,  8948,    27,    91,   408,  8757,   842,    91,
#            397,  2610,   525,   264, 10950,  6888, 25302,    13, 63284,   279,
#           2701, 11972,  2906,  6888,  3491,  3019,   553,  3019,    13,  2411,
#            279,   835,   315,   697,  6291,    11,  1584,   279,  1590,  4226,
#            438,   510,   820,   366,  4082,  1784,    91,    68,   354,   842,
#             91,  1784,    91,  2468,  8757,   842,    91,    29,   872,    27,
#             91,   408,  8757,   842,    91,   397,    32, 62619,  4990,   220,
#             17, 48839,   315,  6303, 23788,   323,  4279,   429,  1753,  4158,
#          23788,    13,   220,  2585,  1657, 48839,   304,  2790,  1558,   432,
#           1896, 75414,    91,    68,   354,   842,    91,  1784,    91,  2468,
#           8757,   842,    91,    29, 77091,    27,    91,   408,  8757,   842,
#             91,   397]], device='cuda:1'), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1]], device='cuda:1')}
# The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
# {'input_ids': tensor([[   27,    91,  7265,  3575,  4326,    91,  1784,    91,  2468,  8757,
#            842,    91,    29,  8948,    27,    91,   408,  8757,   842,    91,
#            397,  2610,   525,   264, 10950,  6888, 25302,    13, 63284,   279,
#           2701, 11972,  2906,  6888,  3491,  3019,   553,  3019,    13,  2411,
#            279,   835,   315,   697,  6291,    11,  1584,   279,  1590,  4226,
#            438,   510,   820,   366,  4082,  1784,    91,    68,   354,   842,
#             91,  1784,    91,  2468,  8757,   842,    91,    29,   872,    27,
#             91,   408,  8757,   842,    91,   397, 18315,   295,   748, 77778,
#          10962,   220,    16,    21, 18805,   817,  1899,    13,  2932, 49677,
#           2326,   369, 17496,  1449,  6556,   323,   293,  2050, 54304,  1330,
#            369,  1059,  4780,  1449,  1899,   448,  3040,    13,  2932, 30778,
#            279, 26313,   518,   279, 20336,     6,  3081,  7298,   369,   400,
#             17,   817,  7722, 35985, 18636,    13,  2585,  1753,   304, 11192,
#           1558,  1340,  1281,  1449,  1899,   518,   279, 20336,     6,  3081,
#          75414,    91,    68,   354,   842,    91,  1784,    91,  2468,  8757,
#            842,    91,    29, 77091,    27,    91,   408,  8757,   842,    91,
#            397]], device='cuda:0'), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], device='cuda:0')}


# The following generation flags are not valid and may be ignored: ['top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
# tensor([[   27,    91,  7265,  3575,  4326,    91,  1784,    91,  2468,  8757,
#            842,    91,    29,  8948,    27,    91,   408,  8757,   842,    91,
#            397,  2610,   525,   264, 10950,  6888, 25302,    13, 63284,   279,
#           2701, 11972,  2906,  6888,  3491,  3019,   553,  3019,    13,  2411,
#            279,   835,   315,   697,  6291,    11,  1584,   279,  1590,  4226,
#            438,   510,   820,   366,  4082,  1784,    91,    68,   354,   842,
#             91,  1784,    91,  2468,  8757,   842,    91,    29,   872,    27,
#             91,   408,  8757,   842,    91,   397,    32, 62619,  4990,   220,
#             17, 48839,   315,  6303, 23788,   323,  4279,   429,  1753,  4158,
#          23788,    13,   220,  2585,  1657, 48839,   304,  2790,  1558,   432,
#           1896, 75414,    91,    68,   354,   842,    91,  1784,    91,  2468,
#           8757,   842,    91,    29, 77091,    27,    91,   408,  8757,   842,
#             91,   397,  1249, 11625,   419,  3491,    11,  1077,   594,  1438,
#            432,  1495,  1119,  7354,  1447,    16,    13,  3070, 10331, 53627,
#          95518,   362]], device='cuda:1')
# None
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:06<00:00,  6.61s/it]
# tensor([[   27,    91,  7265,  3575,  4326,    91,  1784,    91,  2468,  8757,
#            842,    91,    29,  8948,    27,    91,   408,  8757,   842,    91,
#            397,  2610,   525,   264, 10950,  6888, 25302,    13, 63284,   279,
#           2701, 11972,  2906,  6888,  3491,  3019,   553,  3019,    13,  2411,
#            279,   835,   315,   697,  6291,    11,  1584,   279,  1590,  4226,
#            438,   510,   820,   366,  4082,  1784,    91,    68,   354,   842,
#             91,  1784,    91,  2468,  8757,   842,    91,    29,   872,    27,
#             91,   408,  8757,   842,    91,   397, 18315,   295,   748, 77778,
#          10962,   220,    16,    21, 18805,   817,  1899,    13,  2932, 49677,
#           2326,   369, 17496,  1449,  6556,   323,   293,  2050, 54304,  1330,
#            369,  1059,  4780,  1449,  1899,   448,  3040,    13,  2932, 30778,
#            279, 26313,   518,   279, 20336,     6,  3081,  7298,   369,   400,
#             17,   817,  7722, 35985, 18636,    13,  2585,  1753,   304, 11192,
#           1558,  1340,  1281,  1449,  1899,   518,   279, 20336,     6,  3081,
#          75414,    91,    68,   354,   842,    91,  1784,    91,  2468,  8757,
#            842,    91,    29, 77091,    27,    91,   408,  8757,   842,    91,
#            397,  1249,  1477,   700,  1246,  1753, 53665,  3643,  1449,  1899,
#            518,   279, 20336,     6,  3081,    11,   582,  1184,   311, 11047,
#            279]], device='cuda:0')
# None