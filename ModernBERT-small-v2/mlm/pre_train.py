import os
import torch
import math
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

INIT_MODEL_PATH = "./modernbert-small-init"
TRAIN_FILE = "data/combined_mlm_dataset.parquet"
OUTPUT_DIR = "./modernbert-small-mlm"


def run_pretraining():
    # 1. FIND THE ACTUAL LAST CHECKPOINT
    last_checkpoint = None
    if os.path.isdir(OUTPUT_DIR):
        last_checkpoint = get_last_checkpoint(OUTPUT_DIR)
        if last_checkpoint:
            print(f"--- Found Actual Last Checkpoint: {last_checkpoint} ---")

    # 2. LOAD MODEL
    # On T4, we MUST load in float16 if we intend to use fp16=True to avoid the Scaler error.
    print(f"--- Loading Model for Resumption ---")
    model = AutoModelForMaskedLM.from_pretrained(
        last_checkpoint if last_checkpoint else INIT_MODEL_PATH,
        dtype=torch.float32,  # For stability; training done in f16
        attn_implementation="sdpa",
    )

    # Ensure weights are tied (fixes the 'decoder.weight' missing warning)
    model.tie_weights()

    if torch.cuda.is_available():
        model = model.to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(INIT_MODEL_PATH)

    # 3. DATASET
    print("--- Preparing Dataset ---")
    dataset = load_dataset("parquet", data_files={"train": TRAIN_FILE})

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=1024,  # Match your 512*2 setting
            return_special_tokens_mask=True,
            return_token_type_ids=False,
        )

    tokenized_datasets = dataset["train"].map(
        tokenize_function, batched=True, remove_columns=["text"], num_proc=4
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.30
    )

    # 4. TRAINING ARGUMENTS
    # We update save_steps to 2500 to match the checkpoint state
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        fp16=True,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        learning_rate=5e-4,
        lr_scheduler_type="cosine",
        warmup_steps=1000,
        weight_decay=0.01,
        logging_steps=50,
        save_steps=2500,  # Matched to checkpoint
        save_total_limit=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=data_collator,
    )

    # 5. EXECUTION
    print("--- Resuming Student Pre-training ---")
    # Using the automatically detected last_checkpoint
    train_result = trainer.train()

    # 6. SAVE

    trainer.save_model(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")


if __name__ == "__main__":
    run_pretraining()
