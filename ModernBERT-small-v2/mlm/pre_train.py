"""
Phase 1A: MLM Pre-training at 1,024-token context length.

Trains the GUIDE-initialized student (./models/base-model) on the full dataset
with frozen tok_embeddings and saves the checkpoint to ./modernbert-small-mlm/final.
If downstream chunks exceed 1,024 tokens, optionally run pre_train_phase1b_optional.py
after this script completes.
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

INIT_MODEL_PATH = "./models/base-model"
TRAIN_FILE = "data/combined_mlm_dataset.parquet"
OUTPUT_DIR = "./modernbert-small-mlm"


def run_pretraining():
    # 1. FIND THE ACTUAL LAST CHECKPOINT
    last_checkpoint = None
    if os.path.isdir(OUTPUT_DIR):
        last_checkpoint = get_last_checkpoint(OUTPUT_DIR)
        if last_checkpoint:
            print(f"--- Found Actual Last Checkpoint: {last_checkpoint} ---")

    # 2. LOAD MODEL (untied so decoder.weight trains while tok_embeddings stays frozen)
    print("--- Loading Model ---")
    load_path = last_checkpoint if last_checkpoint else INIT_MODEL_PATH
    model = AutoModelForMaskedLM.from_pretrained(
        load_path,
        dtype=torch.float32,  # Master weights in fp32; mixed-precision training in fp16
        attn_implementation="sdpa",
        tie_word_embeddings=False,
    )

    # On a fresh start from INIT_MODEL_PATH, warm-start the untied MLM decoder from tok_embeddings
    if not last_checkpoint and hasattr(model, "decoder"):
        with torch.no_grad():
            model.decoder.weight.copy_(
                model.model.embeddings.tok_embeddings.weight.detach().clone()
            )

    # Freeze only the token embedding lookup table (keep embeddings.norm & decoder trainable)
    model.model.embeddings.tok_embeddings.weight.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"--- Parameters | Trainable: {trainable:,} | Frozen: {frozen:,} ---")

    tokenizer = AutoTokenizer.from_pretrained(INIT_MODEL_PATH)

    # 3. DATASET
    print("--- Preparing Dataset ---")
    dataset = load_dataset("parquet", data_files={"train": TRAIN_FILE})

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=1024,
            return_special_tokens_mask=True,
            return_token_type_ids=False,
        )

    tokenized_datasets = dataset["train"].map(
        tokenize_function, batched=True, remove_columns=["text"], num_proc=6
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.30
    )

    # 4. TRAINING ARGUMENTS
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        fp16=False,
        bf16=True,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },  # Required with frozen inputs
        learning_rate=5e-4,
        lr_scheduler_type="cosine",
        warmup_steps=1000,
        weight_decay=0.01,
        logging_steps=50,
        save_steps=2500,
        save_total_limit=4,
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=data_collator,
    )

    # 5. EXECUTION
    print("--- Starting / Resuming Student Pre-training ---")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # 6. SAVE
    trainer.save_model(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")


if __name__ == "__main__":
    run_pretraining()
