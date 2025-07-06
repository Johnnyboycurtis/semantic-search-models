# ==============================================================================
#           Training a ModernBERT Model on a Combined Dataset
# ==============================================================================
#
# PURPOSE:
# This script fine-tunes a custom ModernBERT model on a combined dataset of
# AllNLI and MS MARCO triplets. This approach aims to build a more robust
# model by training on both general semantic similarity and retrieval-focused data.
#
# ==============================================================================

import torch
import logging
from datetime import datetime
from sentence_transformers import (
    models,
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
    LoggingHandler,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import (
    TripletEvaluator,
    EmbeddingSimilarityEvaluator,
    SequentialEvaluator,
    SimilarityFunction,
)
from datasets import load_dataset, concatenate_datasets

# --- Configuration ---
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

# --- Step 1: Initialize Our Model ---
# This script assumes you have already run the `build_and_save...` script
# to create a blank model architecture.
model_path = "ModernBERT-small/distilled-kldiv-ModernBERT-small/checkpoint-2266"
logging.info(f"Loading custom blank model architecture from: {model_path}")

word_embedding_model = models.Transformer(model_path)
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode='mean'
)

model = SentenceTransformer(
    modules=[word_embedding_model, pooling_model],
    model_card_data=SentenceTransformerModelCardData(
        language="en",
        license="apache-2.0",
        model_name="ModernBERT-medium-NLI-MSMARCO",
    ),
    model_kwargs={"torch_dtype": torch.bfloat16}
)
logging.info("SUCCESS: Blank ModernBERT model loaded into a SentenceTransformer wrapper.")


# --- Step 2: Prepare the Datasets ---
# We will combine two different triplet datasets to create a richer training set.

# --- Dataset 1: AllNLI ---
logging.info("Loading dataset 'sentence-transformers/all-nli'...")
nli_train_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="train").shuffle(123).select(range(10000))
eval_dataset_nli = load_dataset("sentence-transformers/all-nli", "triplet", split="dev")


# --- Dataset 2: TriviaQA ---
logging.info("Loading dataset 'sentence-transformers/msmarco-msmarco-distilbert-base-v3'...")
trivia_qa_dataset = load_dataset("sentence-transformers/trivia-qa-triplet", "triplet", split="train").shuffle(123).select(range(10000))
# This dataset uses 'query' as the anchor, so we rename it to match 'all-nli'

# --- Dataset 2: MS MARCO ---
logging.info("Loading dataset 'sentence-transformers/msmarco-msmarco-distilbert-base-v3'...")
msmarco_dataset = load_dataset("sentence-transformers/msmarco-msmarco-distilbert-base-v3", "triplet", split="train").shuffle(123).select(range(100000))
# This dataset uses 'query' as the anchor, so we rename it to match 'all-nli'
msmarco_dataset = msmarco_dataset.rename_column("query", "anchor")
msmarco_splits = msmarco_dataset.train_test_split(test_size=5000, seed=42)
msmarco_train_dataset = msmarco_splits["train"]
eval_dataset_msmarco = msmarco_splits["test"]
del msmarco_dataset


# --- Concatenate Datasets ---
logging.info("Concatenating datasets...")
# Combine the training sets into one large dataset.
train_dataset = concatenate_datasets([nli_train_dataset, trivia_qa_dataset, msmarco_train_dataset])
# It's good practice to shuffle the combined dataset.
train_dataset = train_dataset.shuffle(seed=25)
logging.info(f"SUCCESS: Combined training dataset created with {len(train_dataset):,} examples.")


# --- Load STS Evaluation Dataset ---
# We still use STSb as our primary measure of semantic understanding.
logging.info("Loading STS-benchmark dataset for evaluation...")
eval_dataset_stsb = load_dataset("sentence-transformers/stsb", split="validation")


# --- Step 3: Define the Loss Function ---
# MultipleNegativesRankingLoss is perfect for triplet data from both sources.
logging.info("Defining the loss function: MultipleNegativesRankingLoss.")
loss = MultipleNegativesRankingLoss(model)


# --- Step 4: Configure Training Arguments ---
output_dir = "./ModernBERT-small/distilled-kldiv-ModernBERT-small/post_training"
logging.info(f"Training arguments configured. Checkpoints will be saved to: {output_dir}")

args = SentenceTransformerTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1, # One epoch over the large combined dataset is usually sufficient.
    per_device_train_batch_size=32,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    bf16=True, # Use bfloat16 for modern GPUs
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    eval_strategy="steps",
    eval_steps=5000, # Evaluate less frequently on this very large dataset
    save_strategy="steps",
    save_steps=5000,
    save_total_limit=4,
    load_best_model_at_end=True,
    metric_for_best_model= "sts-dev_spearman_cosine", #"eval_all-nli-dev_cosine_accuracy",
    logging_steps=100,
    run_name="modernbert-multi-dataset",
)


# --- Step 5: Set Up the Evaluators ---
logging.info("Setting up evaluators for validation...")

# Evaluator 1: NLI triplet accuracy
nli_evaluator = TripletEvaluator(
    anchors=eval_dataset_nli["anchor"],
    positives=eval_dataset_nli["positive"],
    negatives=eval_dataset_nli["negative"],
    name="all-nli-dev",
)

# Evaluator 2: STS benchmark Spearman correlation
stsb_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=eval_dataset_stsb["sentence1"],
    sentences2=eval_dataset_stsb["sentence2"],
    scores=eval_dataset_stsb["score"],
    main_similarity=SimilarityFunction.COSINE,
    name="sts-dev",
)


# We also use a TripletEvaluator on the NLI dev set.
dev_evaluator_msmarco = TripletEvaluator(
    anchors=eval_dataset_msmarco["anchor"],
    positives=eval_dataset_msmarco["positive"],
    negatives=eval_dataset_msmarco["negative"],
    name="msmarco-dev", # A label for the output logs
)

evaluator = SequentialEvaluator([nli_evaluator, stsb_evaluator, dev_evaluator_msmarco])


# --- Step 6: Initialize and Start the Trainer ---
logging.info("Initializing the SentenceTransformerTrainer...")
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=loss,
    evaluator=evaluator,
)

print("\n" + "="*80)
print("🚀🚀🚀 STARTING TRAINING ON COMBINED DATASET 🚀🚀🚀")
print("="*80 + "\n")

trainer.train()

print("\n" + "="*80)
print("🏁🏁🏁 TRAINING COMPLETE 🏁🏁🏁")
print("="*80 + "\n")


# --- Step 7: Save the Final, Trained Model ---
final_model_path = f"{output_dir}/final"
logging.info(f"Saving the final, best-performing model to: {final_model_path}")
model.save_pretrained(final_model_path)

print(f"\n✅ All done! Your newly trained model is ready at '{final_model_path}'.")
