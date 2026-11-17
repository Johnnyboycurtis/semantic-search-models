"""
OPTIONAL: Phase 1B MLM Context-Length Extension (1,024 -> 4,096 tokens).

When to run this script:
  - ONLY if your downstream retrieval / RAG chunks exceed 1,024 tokens.
  - Run this AFTER Phase 1A (pre_train.py) has finished and saved ./modernbert-small-mlm/final.
  - If your downstream chunks are <= 1,024 tokens, you can skip Phase 1B completely.
"""

import os
import torch
from datasets import load_from_disk
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

# Load from Phase 1A's final checkpoint & pre-tokenized dataset from tokenize-finewiki-en.py
PHASE1A_MODEL_PATH = "./modernbert-small-mlm/final"
TOKENIZED_DATASET_DIR = "/content/drive/MyDrive/data/finewiki_4096_filtered"
OUTPUT_DIR = "./modernbert-small-mlm-4096"

# Context extension settings
MAX_LENGTH = 4096



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
    dtype=torch.float32,  # Master weights in fp32; mixed-precision training in bf16
    attn_implementation="sdpa",
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

# 3. DATASET (Load pre-tokenized & filtered Arrow dataset from tokenize-finewiki-en.py)
print(f"--- Loading Pre-tokenized Dataset from {TOKENIZED_DATASET_DIR} ---")
tokenized_datasets = load_from_disk(TOKENIZED_DATASET_DIR)
print(f"--- Loaded {len(tokenized_datasets):,} long-document rows ---")
splits = tokenized_datasets.train_test_split(test_size=1000, seed=42)
train_dataset = splits["train"]
eval_dataset = splits["test"]

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.30
)

# 4. TRAINING ARGUMENTS
# Tuned for 96GB G4 at 4096 tokens: batch 16 * grad_accum 2 = effective batch 32.
# Run for a short 1-epoch / ~5-10% Phase 1A budget with a lower LR (1e-4).
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    fp16=False,
    bf16=True,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    gradient_checkpointing=False,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    weight_decay=0.01,
    logging_steps=25,
    eval_strategy="steps",
    eval_steps=2500,
    save_steps=2500,
    save_total_limit=3,
    dataloader_num_workers=4,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# 5. EXECUTION
print("--- Starting / Resuming Optional Phase 1B Context Extension ---")
trainer.train(resume_from_checkpoint=last_checkpoint)

# 6. SAVE
trainer.save_model(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")


