# ==============================================================================
#           Training a Small ModernBERT for Sentence Embeddings
# ==============================================================================
#
# PURPOSE:
# This script fine-tunes a custom, small-scale ModernBERT model using a
# multi-task, multi-loss training setup to generate high-quality sentence embeddings.
#

### Loss Function Upgrades

# To further improve model performance and output quality, the training loss functions were upgraded based on the following rationale:

# 1.  **`MultipleNegativesRankingLoss` → `MultipleNegativesSymmetricRankingLoss`**
#     *   **Reasoning:** The original loss function only trains in one direction (e.g., `query → answer`). The symmetric version adds a second, "backward" loss term (`answer → query`). This creates a more robust and versatile semantic understanding, as the model must learn a reciprocal relationship between sentence pairs, leading to a higher-quality embedding space.

# 2.  **`CosineSimilarityLoss` → `CoSENTLoss`**
#     *   **Reasoning:** `CosineSimilarityLoss` can sometimes produce a compressed range of similarity scores. `CoSENTLoss` is a more modern alternative that directly optimizes the relative ranking of all pairs in a batch. This provides a stronger training signal and typically results in better-calibrated similarity scores that are more spread out and intuitive, improving the model's performance on regression-based similarity tasks.
# ==============================================================================

import logging
import datetime
from pathlib import Path
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
    losses,
)
from sentence_transformers.training_args import MultiDatasetBatchSamplers
from sentence_transformers.evaluation import (
    TripletEvaluator,
    EmbeddingSimilarityEvaluator,
    SequentialEvaluator,
    SimilarityFunction,
)
import torch

# --- Basic Setup ---
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
START_TIME = datetime.datetime.now()
logging.info(f"START: {START_TIME}")

# --- Step 1: Initialize Our Model ---
# We load the blank ModernBERT architecture. The SentenceTransformer class
# handles module creation (Transformer + Pooling) automatically.
# We let the trainer handle device placement and performance optimizations.
model_path = "./ModernBERT-small/blank_model"
logging.info(f"Loading custom blank model architecture from: {model_path}")
model = SentenceTransformer(
    model_path,
    model_card_data=SentenceTransformerModelCardData(
        language="en",
        license="apache-2.0",
        model_name="ModernBERT-small for General Purpose Similarity",
    ),
    model_kwargs={
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_2",
        "device_map": "cuda"
    }
)
logging.info("SUCCESS: Blank ModernBERT model loaded into a SentenceTransformer wrapper.")
print(model)


# --- Step 2: Prepare Datasets for Multi-Task Training ---
# We will train on a diverse mix of datasets to make the model robust.
# Each dataset will be mapped to a specific loss function.
logging.info("\nLoading datasets for multi-task training...")

# Existing Datasets
nli_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="train")
quora_dataset = load_dataset("sentence-transformers/quora-duplicates", "triplet", split="train")
natural_questions = load_dataset("sentence-transformers/natural-questions", split="train")
stsb_dataset = load_dataset("sentence-transformers/stsb", split="train")

# New Datasets from the example
sentence_compression_dataset = load_dataset("sentence-transformers/sentence-compression", split="train")
simple_wiki_dataset = load_dataset("sentence-transformers/simple-wiki", split="train")
altlex_dataset = load_dataset("sentence-transformers/altlex", split="train")
coco_captions_dataset = load_dataset("sentence-transformers/coco-captions", split="train")
flickr30k_captions_dataset = load_dataset("sentence-transformers/flickr30k-captions", split="train")
yahoo_answers_dataset = load_dataset("sentence-transformers/yahoo-answers", "title-question-answer-pair", split="train")
stack_exchange_dataset = load_dataset("sentence-transformers/stackexchange-duplicates", "title-title-pair", split="train")


# Combine all training datasets into a single dictionary.
# The keys are important as we'll use them to map to loss functions.
train_dataset = {
    "nli": nli_dataset,
    "quora": quora_dataset,
    "natural_questions": natural_questions,
    "stsb": stsb_dataset,
    "sentence_compression": sentence_compression_dataset,
    "simple_wiki": simple_wiki_dataset,
    "altlex": altlex_dataset,
    "coco_captions": coco_captions_dataset,
    "flickr30k_captions": flickr30k_captions_dataset,
    "yahoo_answers": yahoo_answers_dataset,
    "stack_exchange": stack_exchange_dataset,
}
logging.info(f"Training with {len(train_dataset)} datasets: {list(train_dataset.keys())}")


# --- Step 3: Define Multiple Loss Functions ---

### Loss Function Upgrades

# To further improve model performance and output quality, the training loss functions were upgraded based on the following rationale:

# 1.  **`MultipleNegativesRankingLoss` → `MultipleNegativesSymmetricRankingLoss`**
#     *   **Reasoning:** The original loss function only trains in one direction (e.g., `query → answer`). 
# The symmetric version adds a second, "backward" loss term (`answer → query`). This creates a more robust and versatile semantic understanding, as the model must learn a reciprocal relationship between sentence pairs, leading to a higher-quality embedding space.

# 2.  **`CosineSimilarityLoss` → `CoSENTLoss`**
#     *   **Reasoning:** `CosineSimilarityLoss` can sometimes produce a compressed range of similarity scores. 
# `CoSENTLoss` is a more modern alternative that directly optimizes the relative ranking of all pairs in a batch. This provides a stronger training signal and typically results in better-calibrated similarity scores that are more spread out and intuitive, improving the model's performance on regression-based similarity tasks.

# We create a dictionary that maps each dataset (by its key from the dictionary above)
# to the appropriate loss function.
logging.info("\nDefining multiple loss functions for each dataset type...")

# Loss for triplet and pair datasets (contrastive learning)
# mnrl_loss = losses.MultipleNegativesRankingLoss(model)
mnrl_loss = losses.CachedMultipleNegativesSymmetricRankingLoss(model) # upgrade

# Loss for STS dataset (regression task)
#cosine_loss = losses.CosineSimilarityLoss(model)
cosine_loss = losses.CoSENTLoss(model) # upgrade

# The mapping dictionary. Keys MUST match the keys in `train_dataset`.
# For datasets that are just lists of sentences or pairs, MNSRL can still work
# by treating them as (sentence1, sentence2) pairs or by generating in-batch negatives.
loss_functions = {
    "nli": mnrl_loss,
    "quora": mnrl_loss,
    "natural_questions": mnrl_loss,
    "stsb": cosine_loss, # STS is a regression task, so CoSENTLoss is ideal
    "sentence_compression": mnrl_loss, # (sentence_1, sentence_2) pairs
    "simple_wiki": mnrl_loss, # Single sentences, MNSRL will use in-batch negatives
    "altlex": mnrl_loss, # (sentence1, sentence2) pairs
    "coco_captions": mnrl_loss, # Single sentences, MNSRL will use in-batch negatives
    "flickr30k_captions": mnrl_loss, # Single sentences, MNSRL will use in-batch negatives
    "yahoo_answers": mnrl_loss, # (title, question, answer) can be treated as pairs for MNSRL
    "stack_exchange": mnrl_loss, # (title1, title2) pairs
}



# --- Step 4: Configure Training Arguments ---
# These are the hyperparameters for our training run.
output_dir = f"ModernBERT-small/training-run-{START_TIME.strftime('%Y-%m-%d_%H-%M-%S')}"
logging.info(f"\nTraining arguments configured. Checkpoints will be saved to: {output_dir}")

args = SentenceTransformerTrainingArguments(
    # Required: Where to save checkpoints.
    output_dir=output_dir,

    # --- Key Training Parameters ---
    num_train_epochs=1, # 1 epoch is a strong baseline for large, mixed datasets
    per_device_train_batch_size=32, # Adjust based on your GPU's VRAM
    # Use a higher learning rate for training from scratch, as per the ModernBERT paper
    learning_rate=5e-4,
    warmup_ratio=0.05, # 5% of steps for warmup
    
    # Performance optimizations
    fp16=False, # Set to True if your GPU is not Ampere/Hopper
    bf16=True,  # Set to True for Ampere/Hopper GPUs (A100, H100, RTX 30/40xx)
    bf16_full_eval=True,
    
    # Multi-dataset sampling strategy
    multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL, # round robin?

    # --- Evaluation and Saving Strategy ---
    eval_strategy="steps",
    eval_steps=2000,
    save_strategy="steps",
    save_steps=2000,
    save_total_limit=4,
    
    # Load the best model at the end of training
    load_best_model_at_end=True,
    metric_for_best_model="sts-dev_spearman_cosine",

    # --- Logging and Reporting ---
    logging_steps=500,
    run_name="small-modernbert-multi-task",
)


# --- Step 5: Set Up the Evaluators ---
# We use evaluators to get interpretable metrics on validation sets.
logging.info("\nSetting up evaluators for validation...")

# Load evaluation datasets
eval_nli = load_dataset("sentence-transformers/all-nli", "triplet", split="dev")
eval_stsb = load_dataset("sentence-transformers/stsb", split="validation")

# Evaluator 1: For NLI triplets
nli_evaluator = TripletEvaluator(
    anchors=list(eval_nli["anchor"]),     
    positives=list(eval_nli["positive"]), 
    negatives=list(eval_nli["negative"]), 
    name="all-nli-dev",
)

# Evaluator 2: For STS benchmark
stsb_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=list(eval_stsb["sentence1"]),
    sentences2=list(eval_stsb["sentence2"]),
    scores=list(eval_stsb["score"]),       
    main_similarity=SimilarityFunction.COSINE,
    name="sts-dev",
)

# Combine evaluators to run them sequentially
evaluator = SequentialEvaluator([nli_evaluator, stsb_evaluator])


# --- Step 6: Initialize and Start the Trainer ---
# The trainer brings everything together and handles the training loop.
logging.info("\nInitializing the SentenceTransformerTrainer...")
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=loss_functions, # Pass the dictionary of losses
    evaluator=evaluator,
)

logging.info("🚀🚀🚀 STARTING Pre-Training 🚀🚀🚀")
trainer.train()
logging.info("🏁🏁🏁 Pre-Training COMPLETE 🏁🏁🏁")


# --- Step 7: Save the Final, Trained Model ---
# Since `load_best_model_at_end=True`, the trainer has already loaded the best
# checkpoint. We just need to save it.
final_model_path = Path(output_dir) / "final"
model.save(str(final_model_path))
logging.info(f"\n✅ Final model saved to: {final_model_path}")



# --- Step 8: Evaluate the model performance on the STS Benchmark test dataset ---
logging.info("\nEvaluating final model on STS Benchmark test dataset...")
test_dataset_stsb = load_dataset("sentence-transformers/stsb", split="test")

test_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=list(test_dataset_stsb["sentence1"]),
    sentences2=list(test_dataset_stsb["sentence2"]),
    scores=list(test_dataset_stsb["score"]),        
    main_similarity=SimilarityFunction.COSINE,
    name="sts-test",
)
test_evaluator(model) # This will print the results to the console and logging

END_TIME = datetime.datetime.now()
logging.info(f"START: {START_TIME} | END: {END_TIME} | DURATION: {END_TIME - START_TIME}")