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

# --- 0. Basic Setup and Logging ---
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
START_TIME = datetime.datetime.now()
logging.info(f"Script started at: {START_TIME}")

# Define output directory for model checkpoints and logs
OUTPUT_DIR_BASE = "ModernBERT-small-1.5"
RUN_TIMESTAMP = START_TIME.strftime('%Y-%m-%d_%H-%M-%S')
output_dir = Path(OUTPUT_DIR_BASE) / f"pre-trained"
output_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists
logging.info(f"Training outputs will be saved to: {output_dir}")

# --- Step 1: Initialize Our Model ---
# We load the blank ModernBERT architecture. The SentenceTransformer class
# handles module creation (Transformer + Pooling) automatically.
# We let the trainer handle device placement and performance optimizations.
model_path = "./ModernBERT-small-1.5/blank_model"
logging.info(f"Loading custom blank model architecture from: {model_path}")
model = SentenceTransformer(
    model_path,
    model_card_data=SentenceTransformerModelCardData(
        language="en",
        license="apache-2.0",
        model_name="ModernBERT-small-1.5 for General Purpose Similarity",
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



# --- 3. Define Multiple Loss Functions ---
logging.info("\nDefining multiple loss functions for each dataset type...")

# Loss for triplet and pair datasets (contrastive learning)
# Upgraded from MultipleNegativesRankingLoss for symmetric training and caching.
mnsrl_loss = losses.CachedMultipleNegativesSymmetricRankingLoss(model, mini_batch_size=64)
mnrl_loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=64)

# Loss for STS dataset (regression task)
# Upgraded from CosineSimilarityLoss for better calibrated similarity scores.
cosent_loss = losses.CoSENTLoss(model, )

# Map each dataset key to its appropriate loss function.
# Keys MUST match the keys in `train_dataset`.
# For datasets that are lists of sentences or pairs, MNSL will use in-batch negatives.
loss_functions = {
    "nli": mnsrl_loss,
    "quora": mnsrl_loss,
    "natural_questions": mnrl_loss,
    "stsb": cosent_loss, # STS is a regression task, CoSENTLoss is ideal
    "sentence_compression": mnsrl_loss, # (sentence_1, sentence_2) pairs
    "simple_wiki": mnsrl_loss, # Single sentences, MNSL will use in-batch negatives
    "altlex": mnsrl_loss, # (sentence1, sentence2) pairs
    "coco_captions": mnsrl_loss, # Single sentences, MNSL will use in-batch negatives
    "flickr30k_captions": mnsrl_loss, # Single sentences, MNSL will use in-batch negatives
    "yahoo_answers": mnrl_loss, # (title, question, answer) can be treated as pairs for MNSL
    "stack_exchange": mnsrl_loss, # (title1, title2) pairs
}


# --- 4. Configure Training Arguments ---
# These are the hyperparameters for our training run.
logging.info("Configuring training arguments...")

args = SentenceTransformerTrainingArguments(
    output_dir=str(output_dir), # Required: Where to save checkpoints.

    # --- Key Training Parameters ---
    num_train_epochs=3, # 1 epoch is a strong baseline for large, mixed datasets
    per_device_train_batch_size=128, # Adjust based on your GPU's VRAM
    learning_rate=5e-4, # Higher learning rate for training from scratch (as per ModernBERT paper)
    warmup_ratio=0.05, # 5% of steps for learning rate warmup
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    
    # Performance optimizations
    fp16=False, # Set to True if your GPU does not support bf16
    bf16=True,  # Set to True for Ampere/Hopper GPUs (A100, H100, RTX 30/40xx)
    bf16_full_eval=True, # Use bf16 for evaluation as well
    
    # Multi-dataset sampling strategy
    # PROPORTIONAL samples batches from each dataset with a probability proportional to its size.
    # This ensures larger datasets contribute more training steps, which is generally desired
    # for a diverse set of datasets with varying sizes.
    multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL,

    # --- Evaluation and Saving Strategy ---
    eval_strategy="steps", # Evaluate every 'eval_steps'
    eval_steps=2000,
    save_strategy="steps", # Save checkpoint every 'save_steps'
    save_steps=2000,
    save_total_limit=4, # Keep only the best 4 checkpoints
    
    # Load the best model at the end of training based on validation metric
    load_best_model_at_end=True,
    metric_for_best_model="sts-dev_spearman_cosine", # Metric to determine the "best" model

    # --- Logging and Reporting ---
    logging_steps=500,
    run_name="small-modernbert-multi-task", # Name for logging/tracking tools (e.g., Weights & Biases)
)


# --- 5. Set Up the Evaluators ---
# We use evaluators to get interpretable metrics on validation sets during training.
logging.info("\nSetting up evaluators for validation...")

# Load evaluation datasets
eval_nli = load_dataset("sentence-transformers/all-nli", "triplet", split="dev")
eval_stsb = load_dataset("sentence-transformers/stsb", split="validation")

# Evaluator 1: For NLI triplets (measures triplet accuracy)
nli_evaluator = TripletEvaluator(
    anchors=list(eval_nli["anchor"]),     
    positives=list(eval_nli["positive"]), 
    negatives=list(eval_nli["negative"]), 
    name="all-nli-dev",
)

# Evaluator 2: For STS benchmark (measures semantic similarity correlation)
stsb_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=list(eval_stsb["sentence1"]),
    sentences2=list(eval_stsb["sentence2"]),
    scores=list(eval_stsb["score"]),       
    main_similarity=SimilarityFunction.COSINE, # Use Cosine Similarity for STS
    name="sts-dev",
)

# Combine evaluators to run them sequentially during evaluation steps
evaluator = SequentialEvaluator([nli_evaluator, stsb_evaluator])


# --- 6. Initialize and Start the Trainer ---
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


# --- 7. Save the Final, Trained Model ---
# Since `load_best_model_at_end=True`, the trainer has already loaded the best
# checkpoint based on the validation metric. We just need to save this model.
final_model_save_path = output_dir / "final_model"
model.save(str(final_model_save_path))
logging.info(f"\n✅ Final model saved to: {final_model_save_path}")


# --- 8. Evaluate the model performance on the STS Benchmark test dataset ---
# This provides an unbiased final performance metric.
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

# --- Script End ---
END_TIME = datetime.datetime.now()
logging.info(f"Script finished at: {END_TIME}")
logging.info(f"Total duration: {END_TIME - START_TIME}")
