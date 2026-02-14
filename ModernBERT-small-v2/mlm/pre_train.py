import os
import torch
import math
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

# 1. ENVIRONMENT SETUP
set_seed(42)

# PATHS: Ensure this points to the OUTPUT_PATH from your initialization script
INIT_MODEL_PATH = "./modernbert-small-init"
TRAIN_FILE = "data/combined_mlm_dataset.parquet"
OUTPUT_DIR = "./modernbert-small-mlm"


def run_pretraining():
    # 2. LOAD GUIDE-INITIALIZED MODEL
    print(f"--- Loading GUIDE-Initialized Backbone from {INIT_MODEL_PATH} ---")

    # We load via AutoModelForMaskedLM.
    # It will load the GUIDE weights for the backbone and randomly
    # initialize the LM head and layers 1-11.
    model = AutoModelForMaskedLM.from_pretrained(
        INIT_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,  # Maintaining F32 for stability
        attn_implementation="sdpa",  # for pre-training in F32
        # attn_implementation="flash_attention_2" # Enabled for speed
    )

    # Move to GPU immediately to avoid CPU-based attention errors
    if torch.cuda.is_available():
        model = model.to("cuda")
        print("Model moved to CUDA.")

    tokenizer = AutoTokenizer.from_pretrained(INIT_MODEL_PATH)

    # 3. DATASET PREPARATION (Sentence-Transformers Paragraph Format)
    print("--- Preparing Dataset ---")
    dataset = load_dataset("parquet", data_files={"train": TRAIN_FILE})

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512 * 2,
            return_special_tokens_mask=True,
            return_token_type_ids=False,  # Strict ModernBERT requirement
        )

    tokenized_datasets = dataset["train"].map(
        tokenize_function, batched=True, remove_columns=["text"], num_proc=4
    )

    # 4. COLLATOR (The 30% ModernBERT "Golden Rule")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.30
    )

    # 5. TRAINING ARGUMENTS (Optimized for 384-dim Student)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        # overwrite_output_dir=True,
        num_train_epochs=6,
        # T4 COMPATIBILITY LEVER 1: Standard Mixed Precision
        fp16=True,  # For T4 GPU
        bf16=False,  # For RTX GPU
        # T4 COMPATIBILITY LEVER 2: Memory Management
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,  # Effective Batch Size = 128
        gradient_checkpointing=True,  # Save VRAM
        # T4 COMPATIBILITY LEVER 3: Stability over speed
        torch_compile=False,
        # Optimizer & Schedule
        learning_rate=5e-4,
        lr_scheduler_type="cosine",
        warmup_steps=1000,  # Reduced warmup because Layer 0/Embeddings are already aligned
        weight_decay=0.01,
        # Monitoring
        logging_steps=50,
        save_steps=2000,
        save_total_limit=4,
    )

    # 6. INITIALIZE TRAINER
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=data_collator,
    )

    # 7. CHECK FOR CHECKPOINT
    resume_from_checkpoint = None
    if os.path.isdir(OUTPUT_DIR):
        # Check if there are any "checkpoint-XXXX" folders inside
        checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
        if checkpoints:
            resume_from_checkpoint = "last-checkpoint"
            print(f"--- Found existing checkpoints in {OUTPUT_DIR}. Resuming... ---")

    # 8. EXECUTION
    print("--- Starting Student Pre-training ---")
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # 9. SAVE FINAL BACKBONE
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    metrics = train_result.metrics
    perplexity = math.exp(metrics["train_loss"])
    print(f"Final Student Train Perplexity: {perplexity:.2f}")

    # Perplexity Metric
    metrics = train_result.metrics
    perplexity = math.exp(metrics["train_loss"])
    print(f"Final Student Train Perplexity: {perplexity:.2f}")


if __name__ == "__main__":
    run_pretraining()
