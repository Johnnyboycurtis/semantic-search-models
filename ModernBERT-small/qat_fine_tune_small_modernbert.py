# ==============================================================================
#           State-of-the-Art Fine-Tuning with Multi-Dataset Training
#                         and NanoBEIR Benchmarking
# ==============================================================================
#
# PURPOSE:
# This script fine-tunes a model using a modern multi-dataset strategy and
# evaluates it against real-world information retrieval benchmarks during
# training using the NanoBEIREvaluator. This provides the most accurate
# picture of the model's performance on its end-goal task: retrieval.
#
# WHAT IT DOES:
# 1.  Loads a pre-trained sentence-transformer model.
# 2.  Loads multiple retrieval-focused datasets (MS MARCO, GooAQ, Natural Questions).
# 3.  Organizes them into a dictionary for the trainer.
# 4.  Uses MultipleNegativesRankingLoss for powerful in-batch negative training.
# 5.  Sets up a round-robin batch sampler to ensure fair training across datasets.
# 6.  Initializes the NanoBEIREvaluator to benchmark on MSMARCO and NQ.
# 7.  Benchmarks the model *before* training to establish a baseline.
# 8.  Trains the model, with NanoBEIR providing the primary evaluation metrics.
#
# ==============================================================================

import torch
import logging
from pathlib import Path
from datetime import datetime
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
    LoggingHandler,
)
from sentence_transformers import losses
from sentence_transformers.training_args import MultiDatasetBatchSamplers
from sentence_transformers.evaluation import (
    NanoBEIREvaluator,  # The new, powerful evaluator
)
from datasets import load_dataset
from logging.handlers import RotatingFileHandler

# NEW: Import quanto for Quantization-Aware Training
import quanto 

# --- Configuration ---
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[RotatingFileHandler(filename="fine_tuning_qat.log")], # Changed log file name
)


# --- Step 1: Initialize the Model to be Fine-Tuned ---
model_path = "./ModernBERT-small-1.5/pre_trained/final_model"
# CHANGE: Make output_dir dynamic for unique runs, add QAT suffix
output_dir_base = Path("./ModernBERT-small-1.5/finetuned_qat") 
output_dir = output_dir_base
output_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists

logging.info(f"Loading model to be fine-tuned from: {model_path}")

model = SentenceTransformer(
    model_path,
    model_card_data=SentenceTransformerModelCardData(
        language="en",
        license="apache-2.0",
        model_name="ModernBERT-small-1.5-Retrieval-BEIR-Tuned-QAT", # Updated model name for QAT
    ),
    model_kwargs={
        "torch_dtype": torch.bfloat16, # Keep for mixed-precision training acceleration
        "attn_implementation": "flash_attention_2",
        "device_map": "cuda"
    }
)

# --- NEW: Quantization-Aware Training (QAT) Integration ---
logging.info("Preparing model for Quantization-Aware Training (QAT) with quanto...")
# This call modifies the model in-place to insert "fake" quantization operations.
# Weights and activations will be simulated as 8-bit integers during training.
# You can try `activations=quanto.no_observer` if INT8 activations hurt too much.
quanto.quantize(model, weights=quanto.int8, activations=quanto.int8) 
logging.info("Model prepared for QAT.")
# --- End QAT Integration ---


# --- Step 2: Prepare the Datasets for Multi-Task Learning ---
train_datasets = {}

logging.info("Loading MS MARCO dataset...")
msmarco_dataset = load_dataset("sentence-transformers/msmarco-msmarco-distilbert-base-v3", "triplet-hard", split="train") #.select(range(1000))
msmarco_dataset = msmarco_dataset.rename_columns({"query": "anchor", "positive": "positive", "negative": "negative"})
train_datasets["msmarco"] = msmarco_dataset

logging.info("Loading GooAQ dataset...")
gooaq_dataset = load_dataset("sentence-transformers/gooaq", "pair", split="train") #.select(range(5000))
gooaq_dataset = gooaq_dataset.rename_columns({"question": "anchor", "answer": "positive"})
train_datasets["gooaq"] = gooaq_dataset

logging.info("Loading Natural Questions dataset...")
nq_dataset = load_dataset("sentence-transformers/natural-questions", "pair", split="train") #.select(range(1000))
nq_dataset = nq_dataset.rename_columns({"query": "anchor", "answer": "positive"})
train_datasets["natural_questions"] = nq_dataset

logging.info(f"SUCCESS: Loaded {len(train_datasets)} training datasets: {list(train_datasets.keys())}")


# --- Step 3: Define the Loss Function ---
logging.info("Defining the loss function: MultipleNegativesRankingLoss.")
mnsrl_loss = losses.CachedMultipleNegativesSymmetricRankingLoss(model, mini_batch_size=64)
mnrl_loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=64)

# Map each dataset key to its appropriate loss function.
# Keys MUST match the keys in `train_datasets`.
loss_functions = {
    "msmarco": mnrl_loss,
    "gooaq": mnsrl_loss,
    "natural_questions": mnsrl_loss,
}
logging.info(f"Loss functions defined for datasets: {list(loss_functions.keys())}")



# --- Step 4: Configure Training Arguments ---
logging.info(f"Training arguments configured. Checkpoints will be saved to: {output_dir}")

args = SentenceTransformerTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=256,
    multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL, # PROPORTIONAL samples batches based on dataset size, ensuring larger datasets like MSMARCO contribute more training steps, which is often beneficial for retrieval fine-tuning.
    learning_rate=5e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    bf16=True, # Keep for mixed-precision training acceleration
    bf16_full_eval=True, # Keep for mixed-precision evaluation acceleration
    eval_strategy="steps",
    eval_steps=2000, # Evaluate on BEIR less frequently, as it takes more time
    save_strategy="steps",
    save_steps=2000,
    save_total_limit=6,
    load_best_model_at_end=True,
    metric_for_best_model="eval_NanoBEIR_mean_cosine_ndcg@10",
    logging_steps=500,
    run_name="modernbert-beir-finetune-qat", # Updated run name
)

# The available evaluation metrics are: ['eval_NanoMSMARCO_cosine_accuracy@1', 'eval_NanoMSMARCO_cosine_accuracy@3', 'eval_NanoMSMARCO_cosine_accuracy@5', 'eval_NanoMSMARCO_cosine_accuracy@10', 'eval_NanoMSMARCO_cosine_precision@1', 'eval_NanoMSMARCO_cosine_precision@3', 'eval_NanoMSMARCO_cosine_precision@5', 'eval_NanoMSMARCO_cosine_precision@10', 'eval_NanoMSMARCO_cosine_recall@1', 'eval_NanoMSMARCO_cosine_recall@3', 'eval_NanoMSMARCO_cosine_recall@5', 'eval_NanoMSMARCO_cosine_recall@10', 'eval_NanoMSMARCO_cosine_ndcg@10', 'eval_NanoMSMARCO_cosine_mrr@10', 'eval_NanoMSMARCO_cosine_map@100', 'eval_NanoNQ_cosine_accuracy@1', 'eval_NanoNQ_cosine_accuracy@3', 'eval_NanoNQ_cosine_accuracy@5', 'eval_NanoNQ_cosine_accuracy@10', 'eval_NanoNQ_cosine_precision@1', 'eval_NanoNQ_cosine_precision@3', 'eval_NanoNQ_cosine_precision@5', 'eval_NanoNQ_cosine_precision@10', 'eval_NanoNQ_cosine_recall@1', 'eval_NanoNQ_cosine_recall@3', 'eval_NanoNQ_cosine_recall@5', 'eval_NanoNQ_cosine_recall@10', 'eval_NanoNQ_cosine_ndcg@10', 'eval_NanoNQ_cosine_mrr@10', 'eval_NanoNQ_cosine_map@100', 'eval_NanoHotpotQA_cosine_accuracy@1', 'eval_NanoHotpotQA_cosine_accuracy@3', 'eval_NanoHotpotQA_cosine_accuracy@5', 'eval_NanoHotpotQA_cosine_accuracy@10', 'eval_NanoHotpotQA_cosine_precision@1', 'eval_NanoHotpotQA_cosine_precision@3', 'eval_NanoHotpotQA_cosine_precision@5', 'eval_NanoHotpotQA_cosine_precision@10', 'eval_NanoHotpotQA_cosine_recall@1', 'eval_NanoHotpotQA_cosine_recall@3', 'eval_NanoHotpotQA_cosine_recall@5', 'eval_NanoHotpotQA_cosine_recall@10', 'eval_NanoHotpotQA_cosine_ndcg@10', 'eval_NanoHotpotQA_cosine_mrr@10', 'eval_NanoHotpotQA_cosine_map@100', 'eval_NanoBEIR_mean_cosine_accuracy@1', 'eval_NanoBEIR_mean_cosine_accuracy@3', 'eval_NanoBEIR_mean_cosine_accuracy@5', 'eval_NanoBEIR_mean_cosine_accuracy@10', 'eval_NanoBEIR_mean_cosine_precision@1', 'eval_NanoBEIR_mean_cosine_precision@3', 'eval_NanoBEIR_mean_cosine_precision@5', 'eval_NanoBEIR_mean_cosine_precision@10', 'eval_NanoBEIR_mean_cosine_recall@1', 'eval_NanoBEIR_mean_cosine_recall@3', 'eval_NanoBEIR_mean_cosine_recall@5', 'eval_NanoBEIR_mean_cosine_recall@10', 'eval_NanoBEIR_mean_cosine_ndcg@10', 'eval_NanoBEIR_mean_cosine_mrr@10', 'eval_NanoBEIR_mean_cosine_map@100', 'eval_runtime', 'eval_samples_per_second', 'eval_steps_per_second', 'epoch']

# --- Step 5: Set Up the Evaluator ---
logging.info("Setting up NanoBEIREvaluator for validation on MSMARCO and NQ...")
beir_eval_datasets = ["MSMARCO", "NQ", "HotpotQA"]
evaluator = NanoBEIREvaluator(dataset_names=beir_eval_datasets)

# --- Benchmark the base model *before* training ---
logging.info("--- Evaluating Base Model on BEIR (Before Training) ---")
evaluator(model) # This will evaluate the QAT-prepared model (with fake quantization)
logging.info("------------------------------------------------------")


# --- Step 6: Initialize and Start the Trainer ---
logging.info("Initializing the SentenceTransformerTrainer for multi-dataset training...")
trainer = SentenceTransformerTrainer(
    model=model, # The model passed here is now QAT-prepared
    args=args,
    train_dataset=train_datasets,
    loss=loss_functions,
    evaluator=evaluator,
)

print("\n" + "="*80)
print("🚀🚀🚀 STARTING MULTI-DATASET & BEIR-EVALUATED FINE-TUNING WITH QAT 🚀🚀🚀") # Updated print message
print("="*80 + "\n")

trainer.train()

print("\n" + "="*80)
print("🏁🏁🏁 TRAINING COMPLETE 🏁🏁🏁")
print("="*80 + "\n")

# --- Step 7: Save the Final, Trained Model ---
final_model_path = output_dir / "final_model_qat" # Changed output name to reflect QAT
logging.info(f"Saving the final, best-performing QAT-trained model to: {final_model_path}")
model.save_pretrained(str(final_model_path)) # This saves the QAT-prepared model

# --- Step 8: Final Evaluation After Training ---
logging.info("--- Evaluating Final Model on BEIR (After Training) ---")
# This evaluation will use the QAT-trained model, still with fake quantization.
evaluator(model, output_path=f"{output_dir}/final_model_qat")
logging.info("-----------------------------------------------------")


print(f"\n✅ All done! Your BEIR-tuned QAT model is ready at '{final_model_path}'.")
