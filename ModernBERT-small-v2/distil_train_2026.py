import logging
import torch
import os
from pathlib import Path
from datasets import load_from_disk
from sentence_transformers import (
    SentenceTransformer,
    losses,
    evaluation,
    SimilarityFunction
)
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, NanoBEIREvaluator

# --- Configuration ---
TEACHER_MODEL_PATH = "teacher_model_reduced" 
STUDENT_MODEL_PATH = "./mlm/remote//modernbert-small-mlm/checkpoint-60000"
DATASET_DIR = "datasets/distillation_train_dataset"
OUTPUT_DIR = "ModernBERT-small-distilled-v2"

BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 2
TORCH_DTYPE = torch.float32

# 1. Load Teacher Model (needed for the MSEEvaluator during validation)
print(f"Loading Teacher Model from: {TEACHER_MODEL_PATH}")
teacher_model = SentenceTransformer(
    TEACHER_MODEL_PATH,
    model_kwargs={
        "torch_dtype": TORCH_DTYPE,
        "attn_implementation": "sdpa",
        "device_map": "cuda",
    },
)

# 2. Load Student Model
print(f"Loading Student Model from: {STUDENT_MODEL_PATH}")
student_model = SentenceTransformer(
    STUDENT_MODEL_PATH,
    model_kwargs={
        "torch_dtype": TORCH_DTYPE,
        "attn_implementation": "sdpa",
        "device_map": "cuda",
    },
)

# 3. Load Pre-computed Dataset
print(f"Loading pre-computed dataset from: {DATASET_DIR}")
# 1. Load
train_dataset = load_from_disk("datasets/distillation_train_dataset") #.select(range(10000))

# 2. Rename 'text' to 'sentence'
# While 'text' might work, the documentation you just shared 
# explicitly names the input column 'sentence'.
if "text" in train_dataset.column_names:
    dataset = train_dataset.rename_column("text", "sentence")

# 3. Cast 'label' to Tensors
#train_dataset.set_format(type=None)


# Verify the type now (Should be <class 'list'>)
sample = train_dataset[0]
print(f"Label type: {type(sample['label'])}") 


# 4. Verify 
print(f"Dataset Columns: {train_dataset.column_names}")
# Expected: ['sentence', 'label']
# 4. Prepare Evaluation
print("Preparing Evaluators...")

# NanoBEIR (Retrieval)
retrieval_evaluator = NanoBEIREvaluator(
    dataset_names=["MSMARCO", "HotpotQA"], 
    batch_size=BATCH_SIZE
)

# MSE (Teacher-Student distance)
# Using 2000 samples from your pre-computed data to check if student is converging to teacher
mse_eval_sample = train_dataset.select(range(min(len(train_dataset), 2000)))
dev_evaluator_mse = evaluation.MSEEvaluator(
    source_sentences=mse_eval_sample["text"],
    target_sentences=mse_eval_sample["text"],
    teacher_model=teacher_model,
    name="mse-dev"
)

dev_evaluator = evaluation.SequentialEvaluator([
    dev_evaluator_mse,
    retrieval_evaluator
])

# 5. Define Training Logic
train_loss = losses.MSELoss(model=student_model)

args = SentenceTransformerTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    warmup_steps=0.1,
    bf16=False, 
    fp16=True,
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=2,
    logging_steps=100,
    run_name="modernbert-distillation",
    load_best_model_at_end=True,
    metric_for_best_model="eval_NanoMSMARCO_cosine_ndcg@10", 
)

# 6. Start Training
trainer = SentenceTransformerTrainer(
    model=student_model,
    args=args,
    train_dataset=train_dataset,
    loss=train_loss,
    evaluator=dev_evaluator,
)

print("Starting training...")
trainer.train()

# 7. Final Save
print(f"Saving final model to {OUTPUT_DIR}/final")
student_model.save(f"{OUTPUT_DIR}/final")