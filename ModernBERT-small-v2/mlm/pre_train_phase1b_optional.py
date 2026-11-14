"""
OPTIONAL: Phase 1B MLM Context-Length Extension (1,024 -> 4,096 tokens).

When to run this script:
  - ONLY if your downstream retrieval / RAG chunks exceed 1,024 tokens.
  - Run this AFTER Phase 1A (pre_train.py) has finished and saved ./modernbert-small-mlm/final.
  - If your downstream chunks are <= 1,024 tokens, you can skip Phase 1B completely.
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

set_seed(42)

# Load from Phase 1A's final checkpoint
PHASE1A_MODEL_PATH = "./modernbert-small-mlm/final"
TRAIN_FILE = "data/combined_mlm_dataset.parquet"
OUTPUT_DIR = "./modernbert-small-mlm-4096"

# Context extension settings
MAX_LENGTH = 4096
MIN_DOC_TOKENS = 1500  # Filter out short snippets to avoid padding waste up to 4096


def run_phase1b():
    # 1. FIND THE ACTUAL LAST CHECKPOINT (for resumption if interrupted)
    last_checkpoint = None
    if os.path.isdir(OUTPUT_DIR):
        last_checkpoint = get_last_checkpoint(OUTPUT_DIR)
        if last_checkpoint:
            print(f"--- Found Actual Last Checkpoint: {last_checkpoint} ---")

    # 2. LOAD PHASE 1A MODEL & EXTEND RoPE CONTEXT
    # ModernBERT uses RoPE (no fixed position table), so we extend max_position_embeddings
    # to 4096 and increase global_rope_theta to 160000.0 for global attention layers (0, 3, 6, 9).
    # local_rope_theta stays at 10000.0 because local layers keep a 128-token sliding window.
    print("--- Loading Phase 1A Model for Context Extension (4096) ---")
    load_path = last_checkpoint if last_checkpoint else PHASE1A_MODEL_PATH
    model = AutoModelForMaskedLM.from_pretrained(
        load_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        tie_word_embeddings=False,
        max_position_embeddings=MAX_LENGTH,
        global_rope_theta=160000.0,
    )

    # Keep token embeddings frozen to preserve the Phase 1A/GUIDE subspace
    model.model.embeddings.tok_embeddings.weight.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"--- Parameters | Trainable: {trainable:,} | Frozen: {frozen:,} ---")

    tokenizer = AutoTokenizer.from_pretrained(PHASE1A_MODEL_PATH)

    # 3. DATASET (Filter strictly to long documents)
    print("--- Preparing & Filtering Long-Document Dataset ---")
    dataset = load_dataset("parquet", data_files={"train": TRAIN_FILE})

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            return_special_tokens_mask=True,
            return_token_type_ids=False,
        )

    tokenized_datasets = dataset["train"].map(
        tokenize_function, batched=True, remove_columns=["text"], num_proc=4
    )

    # Keep only documents that actually need long-context attention (>= 1,500 tokens)
    before_count = len(tokenized_datasets)
    tokenized_datasets = tokenized_datasets.filter(
        lambda x: len(x["input_ids"]) >= MIN_DOC_TOKENS, num_proc=4
    )
    print(
        f"--- Filtered to Long Docs (>= {MIN_DOC_TOKENS} tokens): "
        f"{len(tokenized_datasets):,} / {before_count:,} ---"
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.30
    )

    # 4. TRAINING ARGUMENTS
    # At 4096 tokens on a ~7.62 GiB GPU, per_device_train_batch_size=2 (or 1) avoids OOM;
    # gradient_accumulation_steps=16 keeps effective batch size at 32 (32 * 4096 = 131k tokens/step).
    # Run for a short 1-epoch / ~5-10% Phase 1A budget with a lower LR (1e-4).
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        fp16=False,
        bf16=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },  # Required with frozen inputs
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=25,
        save_steps=500,
        save_total_limit=3,
        dataloader_num_workers=4,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=data_collator,
    )

    # 5. EXECUTION
    print("--- Starting / Resuming Optional Phase 1B Context Extension ---")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # 6. SAVE
    trainer.save_model(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")


if __name__ == "__main__":
    run_phase1b()

