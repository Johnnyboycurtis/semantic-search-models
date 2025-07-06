# ==============================================================================
#                  Fine-Tuning a Model on the STS Benchmark
# ==============================================================================
#
# PURPOSE:
# This script (train_sts.py) takes a pre-trained sentence embedding model
# and fine-tunes it directly on the Semantic Textual Similarity (STS) benchmark
# dataset. This is a regression task where the model learns to predict the
# similarity score (from 0.0 to 1.0) between two sentences.
#
# This is an alternative to contrastive training (like with NLI or MS MARCO)
# and can be very effective for producing high-quality general-purpose embeddings.
#
# WHAT IT DOES:
# 1.  Loads a strong baseline model (your best NLI-trained model).
# 2.  Loads the STSb dataset, which contains sentence pairs and a similarity score.
# 3.  Uses `CosineSimilarityLoss`, which is designed for this regression task.
# 4.  Uses the `SentenceTransformerTrainer` to run the training loop.
# 5.  Includes an `EmbeddingSimilarityEvaluator` to monitor performance.
#
# ==============================================================================

import logging
from datetime import datetime
import torch
from datasets import load_dataset

from sentence_transformers import (
    LoggingHandler,
    SentenceTransformer,
    losses,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

# --- Configuration ---
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

# --- Step 1: Load the Model to be Fine-Tuned ---
# We start with our best model so far: the "medium" model that was
# fine-tuned on the AllNLI dataset. This provides a strong foundation.
# Ensure this path points to your best NLI-trained model.
# model_path = './ModernBERT-small/training-small-modernbert/final'
model_path = "ModernBERT-small/distilled-kldiv-ModernBERT-small/checkpoint-2266"

# Define where we will save the final, STS-tuned model.
#output_dir = "./ModernBERT-small/sts-tuned-modernbert-small"
output_dir = "./ModernBERT-small/distilled-sts-tuned-modernbert-small"

# Training hyperparameters
train_batch_size = 16
num_train_epochs = 4 # STSb is a small dataset, so more epochs are needed

logging.info(f"Loading base model from: {model_path}")
model = SentenceTransformer(model_path, model_kwargs={"torch_dtype": torch.bfloat16})


# --- Step 2: Load the STS Benchmark Dataset ---
# This dataset is perfectly formatted for regression tasks.
dataset_name = "sentence-transformers/stsb"
logging.info(f"Loading dataset: {dataset_name}")

train_dataset = load_dataset(dataset_name, split="train")
eval_dataset = load_dataset(dataset_name, split="validation")
test_dataset = load_dataset(dataset_name, split="test")

logging.info(f"Train dataset size: {len(train_dataset)}")
logging.info(f"Validation dataset size: {len(eval_dataset)}")
logging.info(f"Test dataset size: {len(test_dataset)}")


# --- Step 3: Define the Loss Function ---
# `CosineSimilarityLoss` is designed for this task. It takes two sentences
# and a score between 0 and 1. It fine-tunes the model so that the
# cosine similarity between the sentence embeddings matches the gold score.
# The scores in STSb (0-5) are automatically normalized to 0-1 by the loss function.
train_loss = losses.CosineSimilarityLoss(model=model)
logging.info("Using CosineSimilarityLoss")


# --- Step 4: Set up the Evaluator ---
# The `EmbeddingSimilarityEvaluator` is the perfect counterpart to the loss function.
# It computes the Spearman correlation between the model's predicted cosine
# similarity and the gold standard scores.
evaluator = EmbeddingSimilarityEvaluator(
    sentences1=eval_dataset["sentence1"],
    sentences2=eval_dataset["sentence2"],
    scores=eval_dataset["score"],
    name="sts-dev"
)


# --- Step 5: Configure and Run the Trainer ---
args = SentenceTransformerTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=train_batch_size,
    warmup_ratio=0.1,
    fp16=False,
    bf16=True, # Recommended for modern GPUs
    learning_rate=2e-5,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    logging_steps=100,
    metric_for_best_model="sts-dev_spearman_cosine",
    load_best_model_at_end=True,
    run_name="sts-finetune",
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    evaluator=evaluator,
    loss=train_loss,
)

logging.info("🚀🚀🚀 STARTING STS FINE-TUNING 🚀🚀🚀")
trainer.train()
logging.info("🏁🏁🏁 FINE-TUNING COMPLETE 🏁🏁🏁")

# --- Step 6: Evaluate on the Test Set ---
logging.info("Evaluating on the STS test set...")
test_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=test_dataset["sentence1"],
    sentences2=test_dataset["sentence2"],
    scores=test_dataset["score"],
    name="sts-test"
)
test_evaluator(model)


# --- Step 7: Save the Final Model ---
final_output_dir = f"{output_dir}/final"
model.save(final_output_dir)
logging.info(f"Final STS-tuned model saved to: {final_output_dir}")
