# ==============================================================================
#           Improving a Model with Advanced Knowledge Distillation (KLDivLoss)
# ==============================================================================
#
# PURPOSE:
# This script uses a more advanced knowledge distillation technique with the
# DistillKLDivLoss. Instead of forcing the student to mimic the teacher's exact
# embeddings (like with MSELoss), this trains the student to replicate the
# teacher's *similarity score distribution* across positive and negative pairs.
# This often leads to better performance as it focuses on the relative rankings.
#
# WHAT IT DOES:
# 1.  Loads a strong foundational student model and a powerful teacher model.
# 2.  Loads a triplet dataset (AllNLI).
# 3.  Pre-computes the teacher's similarity scores for each triplet to create the "labels".
# 4.  Uses `DistillKLDivLoss` to train the student.
# 5.  Evaluates performance on STSb and NLI dev sets.
#
# ==============================================================================

import logging
from datetime import datetime
import torch
from datasets import load_dataset, concatenate_datasets
import pandas as pd

from sentence_transformers import (
    LoggingHandler,
    SentenceTransformer,
    losses,
    evaluation,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, TripletEvaluator
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

# --- Configuration ---
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

# --- Step 1: Define Teacher and Student Models ---
# The "teacher" model should be our best-performing model.
teacher_model_name = "BAAI/bge-base-en-v1.5"
# The "student" model is our best foundational model.
student_model_path = './ModernBERT-small/training-small-modernbert/final'

# Define where we will save the final, distilled model.
output_dir = "ModernBERT-small/distilled-kldiv-ModernBERT-small" #+ datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Training hyperparameters
train_batch_size = 64
inference_batch_size = 64 # For the teacher model's encoding step

logging.info(f"Teacher model: {teacher_model_name}")
logging.info(f"Student model (initial): {student_model_path}")

# Load models with bfloat16 for performance gains on compatible hardware
model_kwargs = {"torch_dtype": torch.bfloat16}
teacher_model = SentenceTransformer(teacher_model_name, model_kwargs=model_kwargs)
student_model = SentenceTransformer(student_model_path, model_kwargs=model_kwargs)


# --- Step 2: Load and Prepare Datasets ---
# We use a triplet dataset for this loss function.
logging.info("Loading AllNLI dataset...")

# --- Dataset 1: AllNLI ---
print("\nINFO: Loading dataset 'sentence-transformers/all-nli'...")
nli_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="train").select(range(20000))
eval_dataset_nli = load_dataset("sentence-transformers/all-nli", "triplet", split="dev")

# --- Dataset 2: TriviaQA ---
print("INFO: Loading dataset 'sentence-transformers/trivia-qa-triplet'...")
trivia_qa_dataset = load_dataset("sentence-transformers/trivia-qa-triplet", "triplet", split="train").select(range(30000))
# This dataset uses 'query' as the anchor, so we rename it to match 'all-nli'


# --- Dataset 2: MS MARCO ---
print("Loading dataset 'sentence-transformers/msmarco-msmarco-distilbert-base-v3'...")
msmarco_dataset = load_dataset("sentence-transformers/msmarco-msmarco-distilbert-base-v3", "triplet", split="train").select(range(100000))
# This dataset uses 'query' as the anchor, so we rename it to match 'all-nli'
msmarco_dataset = msmarco_dataset.rename_column("query", "anchor")
msmarco_splits = msmarco_dataset.train_test_split(test_size=5000, seed=42)
msmarco_train_dataset = msmarco_splits["train"]
eval_dataset_msmarco = msmarco_splits["test"]
del msmarco_dataset


# --- Concatenate Datasets ---
print("INFO: Concatenating datasets...")
# Combine the training sets into one large dataset.
train_dataset = concatenate_datasets([nli_dataset, trivia_qa_dataset, msmarco_train_dataset])
# You can shuffle the combined dataset if you want, which is good practice.
train_dataset = train_dataset.shuffle(seed=7936)
print(f"SUCCESS: Combined training dataset created with {len(train_dataset):,} examples.")

logging.info(f"Training dataset size: {len(train_dataset):,}")


# --- Step 3: Pre-compute Teacher Similarity Scores ---
# This is the most important step for this loss. Instead of encoding single
# sentences, we encode the triplets and compute the teacher's similarity scores.
# These scores become the "soft labels" for our student.
logging.info("Mapping dataset with teacher similarity scores... (This may take a while)")

def compute_teacher_scores(batch):
    queries = batch["anchor"]
    positives = batch["positive"]
    negatives = batch["negative"]
    
    # Encode all texts with the teacher model
    emb_queries = teacher_model.encode(queries, batch_size=inference_batch_size)
    emb_positives = teacher_model.encode(positives, batch_size=inference_batch_size)
    emb_negatives = teacher_model.encode(negatives, batch_size=inference_batch_size)

    # Calculate the similarity scores
    pos_scores = teacher_model.similarity_pairwise(emb_queries, emb_positives)
    neg_scores = teacher_model.similarity_pairwise(emb_queries, emb_negatives)

    # The label for DistillKLDivLoss is a tensor of [positive_score, negative_score]
    return {"label": torch.stack([pos_scores, neg_scores], dim=1).tolist()}

train_dataset = train_dataset.map(compute_teacher_scores, batched=True, batch_size=512)


# --- Step 4: Set up Evaluators and Loss ---
# We still use STSb to measure the "real-world" performance.
stsb_eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")
dev_evaluator_stsb = EmbeddingSimilarityEvaluator(
    sentences1=stsb_eval_dataset["sentence1"],
    sentences2=stsb_eval_dataset["sentence2"],
    scores=stsb_eval_dataset["score"],
    name="sts-dev"
)

# We also use a TripletEvaluator on the NLI dev set.
dev_evaluator_nli = TripletEvaluator(
    anchors=eval_dataset_nli["anchor"],
    positives=eval_dataset_nli["positive"],
    negatives=eval_dataset_nli["negative"],
    name="all-nli-dev", # A label for the output logs
)


# We also use a TripletEvaluator on the NLI dev set.
dev_evaluator_msmarco = TripletEvaluator(
    anchors=eval_dataset_msmarco["anchor"],
    positives=eval_dataset_msmarco["positive"],
    negatives=eval_dataset_msmarco["negative"],
    name="msmarco-dev", # A label for the output logs
)



evaluator = evaluation.SequentialEvaluator([dev_evaluator_stsb, dev_evaluator_nli, dev_evaluator_msmarco])

# Define the distillation loss function
train_loss = losses.DistillKLDivLoss(model=student_model)
logging.info("Using DistillKLDivLoss")


# --- Step 5: Configure and Run the Trainer ---
args = SentenceTransformerTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1, # 1 epoch is usually enough for distillation
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=train_batch_size,
    warmup_ratio=0.1,
    fp16=False,
    bf16=True,
    learning_rate=5e-5, # Can often use a slightly higher LR for distillation
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=4,
    logging_steps=100,
    metric_for_best_model="eval_msmarco-dev_cosine_accuracy", #"eval_msmarco-dev_cosine_accuracy", #"sts-dev_spearman_cosine",
    load_best_model_at_end=True,
    run_name="distill-kldiv-modernbert",
)

trainer = SentenceTransformerTrainer(
    model=student_model,
    args=args,
    train_dataset=train_dataset,
    evaluator=evaluator,
    loss=train_loss,
)

logging.info("🚀🚀🚀 STARTING KNOWLEDGE DISTILLATION (KLDivLoss) 🚀🚀🚀")
trainer.train()
logging.info("🏁🏁🏁 DISTILLATION COMPLETE 🏁🏁🏁")

# Save the final, best-performing model
final_output_dir = f"{output_dir}/final"
student_model.save(final_output_dir)
logging.info(f"Final distilled model saved to: {final_output_dir}")
