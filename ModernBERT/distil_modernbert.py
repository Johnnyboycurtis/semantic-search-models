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
model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_2",
        "device_map": "cuda"
    }
teacher_model = SentenceTransformer(teacher_model_name, model_kwargs={"torch_dtype": torch.bfloat16,})
student_model = SentenceTransformer(student_model_path, model_kwargs=model_kwargs)


# --- Step 2: Load and Prepare Datasets ---
# We use triplet and positive-pair datasets for this loss function.
logging.info("Loading datasets for distillation...")

# --- Dataset 1: AllNLI (Triplets) ---
print("\nINFO: Loading dataset 'sentence-transformers/all-nli'...")
# Using a larger subset for distillation if resources allow
nli_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="train").select(range(100000)) # Increased from 20k
eval_dataset_nli = load_dataset("sentence-transformers/all-nli", "triplet", split="dev")

# --- Dataset 2: TriviaQA (Triplets) ---
print("INFO: Loading dataset 'sentence-transformers/trivia-qa-triplet'...")
# Using a larger subset for distillation if resources allow
trivia_qa_dataset = load_dataset("sentence-transformers/trivia-qa-triplet", "triplet", split="train") # Increased from 30k
trivia_qa_dataset = trivia_qa_dataset #.rename_column("query", "anchor") # Ensure consistency

# --- Dataset 3: MS MARCO (Triplets) ---
print("Loading dataset 'sentence-transformers/msmarco-msmarco-distilbert-base-v3'...")
# Using a larger subset for distillation if resources allow
msmarco_dataset = load_dataset("sentence-transformers/msmarco-msmarco-distilbert-base-v3", "triplet", split="train").select(range(300000)) # Increased from 110k
msmarco_dataset = msmarco_dataset.rename_column("query", "anchor")
msmarco_splits = msmarco_dataset.train_test_split(test_size=10000, seed=42)
msmarco_train_dataset = msmarco_splits["train"]
eval_dataset_msmarco = msmarco_splits["test"]
del msmarco_dataset

# --- NEW Dataset 4: Quora Duplicates (Triplets) ---
print("INFO: Loading dataset 'sentence-transformers/quora-duplicates'...")
quora_dataset = load_dataset("sentence-transformers/quora-duplicates", "triplet", split="train").select(range(100000)) # Sample a subset
# This dataset is already in 'anchor', 'positive', 'negative' format

# --- NEW Dataset 5: GooAQ (Positive Pairs for in-batch negatives) ---
print("INFO: Loading dataset 'sentence-transformers/gooaq'...")
gooaq_dataset = load_dataset("sentence-transformers/gooaq", split="train").select(range(200000)) # Sample a subset
gooaq_dataset = gooaq_dataset.rename_columns({"question": "anchor", "answer": "positive"})

# --- NEW Dataset 6: ServiceNow/repliqa (Positive Pairs for in-batch negatives) ---
# Assuming 'repliqa' has 'question' and 'answer' columns for positive pairs
# You'll need to load this from your local path or Hugging Face if it's public
# print("INFO: Loading dataset 'ServiceNow/repliqa'...")
# repliqa_dataset = load_dataset("ServiceNow/repliqa", split="train").select(range(YOUR_DESIRED_SIZE))
# repliqa_dataset = repliqa_dataset.rename_columns({"question": "anchor", "answer": "positive"})


# --- Concatenate All Training Datasets ---
print("INFO: Concatenating all training datasets...")
train_dataset = concatenate_datasets([
    nli_dataset,
    trivia_qa_dataset,
    msmarco_train_dataset,
    quora_dataset,
    gooaq_dataset,
    # repliqa_dataset, # Uncomment if you add repliqa
])
train_dataset = train_dataset.shuffle(seed=7936)
print(f"SUCCESS: Combined training dataset created with {len(train_dataset):,} examples.")
logging.info(f"Total training dataset size: {len(train_dataset):,}")

# --- IMPORTANT: Filter out None or empty string values from the dataset ---
logging.info("Filtering out examples with None or empty strings in text columns...")
original_length = len(train_dataset)
train_dataset = train_dataset.filter(
    lambda example: example["anchor"] is not None and example["anchor"].strip() != "" and
                    example["positive"] is not None and example["positive"].strip() != "" and
                    example["negative"] is not None and example["negative"].strip() != ""
)
filtered_length = len(train_dataset)
if original_length != filtered_length:
    logging.warning(f"Filtered {original_length - filtered_length} examples containing None or empty text fields.")
logging.info(f"Training dataset size after filtering: {len(train_dataset):,}")


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

# --- Evaluator 1: STS-b ---
stsb_eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")
# Convert columns to lists before passing them to the evaluator
stsb_sentences1 = list(stsb_eval_dataset["sentence1"])
stsb_sentences2 = list(stsb_eval_dataset["sentence2"])
stsb_scores = list(stsb_eval_dataset["score"])

dev_evaluator_stsb = EmbeddingSimilarityEvaluator(
    sentences1=stsb_sentences1,
    sentences2=stsb_sentences2,
    scores=stsb_scores,
    name="sts-dev"
)

# --- Evaluator 2: NLI Triplets ---
# Convert columns to lists before passing them to the evaluator
nli_anchors = list(eval_dataset_nli["anchor"])
nli_positives = list(eval_dataset_nli["positive"])
nli_negatives = list(eval_dataset_nli["negative"])

dev_evaluator_nli = TripletEvaluator(
    anchors=nli_anchors,
    positives=nli_positives,
    negatives=nli_negatives,
    name="all-nli-dev",
)

# --- Evaluator 3: MS MARCO Triplets ---
# Convert columns to lists before passing them to the evaluator
msmarco_anchors = list(eval_dataset_msmarco["anchor"])
msmarco_positives = list(eval_dataset_msmarco["positive"])
msmarco_negatives = list(eval_dataset_msmarco["negative"])

dev_evaluator_msmarco = TripletEvaluator(
    anchors=msmarco_anchors,
    positives=msmarco_positives,
    negatives=msmarco_negatives,
    name="msmarco-dev",
)

# Combine the evaluators into a sequence
evaluator = evaluation.SequentialEvaluator([dev_evaluator_stsb, dev_evaluator_nli, dev_evaluator_msmarco])

# --- Step 5: Configure and Run the Trainer ---
# Define the distillation loss function
train_loss = losses.DistillKLDivLoss(model=student_model)
logging.info("Using DistillKLDivLoss")

args = SentenceTransformerTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3, # 1 epoch is usually enough for distillation
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=train_batch_size,
    warmup_ratio=0.1,
    fp16=False,
    bf16=True,
    bf16_full_eval=True,
    learning_rate=5e-5, # Can often use a slightly higher LR for distillation
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=4,
    logging_steps=500,
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