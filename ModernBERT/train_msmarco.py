# ==============================================================================
#           Improving a Model with In-Batch Negative Knowledge Distillation
# ==============================================================================
#
# PURPOSE:
# This script fine-tunes a student model using a powerful in-batch negative
# training setup with MultipleNegativesRankingLoss. This is a form of distillation
# where the student learns to produce a good embedding space for ranking, a skill
# implicitly held by the teacher. Unlike the previous approach, this focuses
# on creating a globally consistent embedding space rather than just replicating
# teacher scores on isolated triplets.
#
# WHAT IT DOES:
# 1.  Loads a strong foundational student model.
# 2.  Loads multiple triplet/pair datasets (AllNLI, TriviaQA, MS MARCO).
# 3.  Uses `MultipleNegativesRankingLoss` which leverages in-batch negatives for
#     highly efficient and effective training. The explicit "negative" column
#     from the dataset is used, but the primary source of negatives comes from
#     other examples in the same batch.
# 4.  Evaluates performance on STSb, NLI, MS MARCO, and a new Paraphrase Mining task.
#
# ==============================================================================

import logging
from datetime import datetime
import torch
from datasets import load_dataset, concatenate_datasets

from sentence_transformers import (
    LoggingHandler,
    SentenceTransformer,
    losses,
    evaluation,
)
# --- CHANGE 1: Import the new evaluator ---
from sentence_transformers.evaluation import (
    EmbeddingSimilarityEvaluator,
    TripletEvaluator,
    ParaphraseMiningEvaluator, # Added this import
)
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

# --- Configuration ---
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

# --- Step 1: Define Student Model ---
# We no longer need the teacher model during this training script
student_model_path = './ModernBERT-small/training-small-modernbert/final'
output_dir = "ModernBERT-small/distilled-mnrl-ModernBERT-small"

# Training hyperparameters
train_batch_size = 64 # MNRL benefits from larger batch sizes if VRAM allows

logging.info(f"Student model (initial): {student_model_path}")

# Load model with bfloat16 for performance gains on compatible hardware
model_kwargs = {"torch_dtype": torch.bfloat16}
student_model = SentenceTransformer(student_model_path, model_kwargs=model_kwargs)


# --- Step 2: Load and Prepare Datasets ---
# We use a triplet dataset, as MultipleNegativesRankingLoss can use it.
# The primary benefit, however, comes from in-batch negatives.
logging.info("Loading datasets...")

# --- Dataset 1: AllNLI ---
nli_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="train").select(range(20000))
eval_dataset_nli = load_dataset("sentence-transformers/all-nli", "triplet", split="dev")

# --- Dataset 2: TriviaQA ---
trivia_qa_dataset = load_dataset("sentence-transformers/trivia-qa-triplet", "triplet", split="train").select(range(30000))

# --- Dataset 3: MS MARCO ---
msmarco_dataset = load_dataset("sentence-transformers/msmarco-msmarco-distilbert-base-v3", "triplet", split="train").select(range(100000))
msmarco_dataset = msmarco_dataset.rename_column("query", "anchor")
msmarco_splits = msmarco_dataset.train_test_split(test_size=5000, seed=42)
msmarco_train_dataset = msmarco_splits["train"]
eval_dataset_msmarco = msmarco_splits["test"]
del msmarco_dataset

# --- Concatenate Datasets ---
train_dataset = concatenate_datasets([nli_dataset, trivia_qa_dataset, msmarco_train_dataset])
train_dataset = train_dataset.shuffle(seed=7936)
logging.info(f"Combined training dataset created with {len(train_dataset):,} examples.")


# --- CHANGE 2: REMOVED a Major Step ---
# --- Step 3: (REMOVED) Pre-compute Teacher Similarity Scores ---
# We are switching to MultipleNegativesRankingLoss, which does not require
# pre-computed teacher scores. It operates directly on the text pairs,
# making the training setup much simpler and faster.
logging.info("Skipping teacher score pre-computation. Using MultipleNegativesRankingLoss instead.")


# --- Step 4: Set up Evaluators and Loss ---
stsb_eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")
dev_evaluator_stsb = EmbeddingSimilarityEvaluator(
    sentences1=stsb_eval_dataset["sentence1"],
    sentences2=stsb_eval_dataset["sentence2"],
    scores=stsb_eval_dataset["score"],
    name="sts-dev"
)

dev_evaluator_nli = TripletEvaluator(
    anchors=eval_dataset_nli["anchor"],
    positives=eval_dataset_nli["positive"],
    negatives=eval_dataset_nli["negative"],
    name="all-nli-dev",
)

dev_evaluator_msmarco = TripletEvaluator(
    anchors=eval_dataset_msmarco["anchor"],
    positives=eval_dataset_msmarco["positive"],
    negatives=eval_dataset_msmarco["negative"],
    name="msmarco-dev",
)

# --- CHANGE 3: ADDED Paraphrase Mining Evaluator ---
logging.info("Adding Paraphrase Mining Evaluator on Quora Duplicates.")
quora_eval_dataset = load_dataset("sentence-transformers/quora-duplicates", "pair", split="train").select(range(5000))


sentences_map = {}
duplicate_pairs = []
for row in quora_eval_dataset:
    # Get the texts from the 'anchor' and 'positive' columns
    sentence1 = row['anchor']
    sentence2 = row['positive']
    
    # Add sentences to the map if they aren't already there
    # This logic is optional but can reduce the size of sentences_map
    if sentence1 not in sentences_map.values():
        s1_id = len(sentences_map)
        sentences_map[s1_id] = sentence1
    else:
        # Find the existing ID
        s1_id = list(sentences_map.keys())[list(sentences_map.values()).index(sentence1)]

    if sentence2 not in sentences_map.values():
        s2_id = len(sentences_map)
        sentences_map[s2_id] = sentence2
    else:
        # Find the existing ID
        s2_id = list(sentences_map.keys())[list(sentences_map.values()).index(sentence2)]
        
    # --- THIS IS THE KEY FIX ---
    # Since every row in the "pair" dataset is a duplicate, we don't need to check a label.
    # We simply add the pair of IDs to our list of duplicates.
    duplicate_pairs.append((s1_id, s2_id))

dev_evaluator_paraphrase = ParaphraseMiningEvaluator(
    sentences_map,
    duplicate_pairs,
    name="quora-paraphrase-dev"
)
# --- END CHANGE 3 ---


evaluator = evaluation.SequentialEvaluator([
    dev_evaluator_stsb,
    dev_evaluator_nli,
    dev_evaluator_msmarco,
    dev_evaluator_paraphrase # Added to the sequence
])


# --- CHANGE 4: SWITCHED the Loss Function ---
# Define the new, more powerful loss function
train_loss = losses.MultipleNegativesRankingLoss(model=student_model)
logging.info("Using MultipleNegativesRankingLoss for in-batch negative training.")
# --- END CHANGE 4 ---


# --- Step 5: Configure and Run the Trainer ---
args = SentenceTransformerTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3, # MNRL is very data-efficient, 1-3 epochs is often enough
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=train_batch_size,
    warmup_ratio=0.1,
    fp16=False,
    bf16=True,
    learning_rate=5e-5,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=4,
    logging_steps=100,
    metric_for_best_model="eval_msmarco-dev_cosine_accuracy",
    load_best_model_at_end=True,
    run_name="distill-mnrl-modernbert",
)

trainer = SentenceTransformerTrainer(
    model=student_model,
    args=args,
    train_dataset=train_dataset,
    evaluator=evaluator,
    loss=train_loss,
)

logging.info("🚀🚀🚀 STARTING IN-BATCH NEGATIVE TRAINING (MNRL) 🚀🚀🚀")
trainer.train()
logging.info("🏁🏁🏁 TRAINING COMPLETE 🏁🏁🏁")

# Save the final, best-performing model
final_output_dir = f"{output_dir}/final"
student_model.save(final_output_dir)
logging.info(f"Final distilled model saved to: {final_output_dir}")