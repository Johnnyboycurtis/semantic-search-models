import logging
import datetime
from pathlib import Path
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.training_args import MultiDatasetBatchSamplers, BatchSamplers # <--- ADDED BatchSamplers
from sentence_transformers.evaluation import (
    TripletEvaluator, 
    SequentialEvaluator,
    EmbeddingSimilarityEvaluator,
    SimilarityFunction
)
import torch

# --- 0. Basic Setup and Logging ---
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
START_TIME = datetime.datetime.now()
logging.info(f"Script started at: {START_TIME}")

# Output config
OUTPUT_DIR_BASE = "ModernBERT-small-2"
output_dir = Path(OUTPUT_DIR_BASE) / "pre-trained"
output_dir.mkdir(parents=True, exist_ok=True)

# --- Step 1: Initialize Blank Model ---
# Note: Ensure this path points to your UNTRAINED architecture skeleton
stage_1_model_path = "./ModernBERT-small-2/modernbert-small-init"

if not Path(stage_1_model_path).exists():
    logging.warning(f"Model path not found at {stage_1_model_path}. Ensure this is correct.")

logging.info(f"Loading Blank Architecture from: {stage_1_model_path}")
model = SentenceTransformer(
    stage_1_model_path,
    model_kwargs={
        "torch_dtype": torch.float32,
        "attn_implementation": "flash_attention_2", 
        "device_map": "cuda"
    }
)
logging.info(model)

# --- Step 2: Prepare Datasets ---
logging.info("\nLoading datasets...")

# 2.1 The "Stabilizers" (General Knowledge)
nli_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="train")
quora_dataset = load_dataset("sentence-transformers/quora-duplicates", "triplet", split="train")
natural_questions = load_dataset("sentence-transformers/natural-questions", split="train")

# 2.2 MSMARCO (The Heavy Hitter)
logging.info("Loading MS MARCO dataset...")
#msmarco_dataset = load_dataset("sentence-transformers/msmarco-msmarco-distilbert-base-v3", "triplet", split="train")
#msmarco_dataset = msmarco_dataset.rename_columns({"query": "anchor", "positive": "positive", "negative": "negative"})
msmarco_dataset = load_dataset("tomaarsen/msmarco-Qwen3-Reranker-0.6B", split="train")

# 2.3 GooAQ (Diversity)
logging.info("Loading GooAQ dataset...")
gooaq_dataset = load_dataset("sentence-transformers/gooaq", "pair", split="train")
# GooAQ is pairs (Query, Answer). We map them to anchor/positive.
gooaq_dataset = gooaq_dataset.rename_columns({"question": "anchor", "answer": "positive"})

# 2.4 Philosophical STS (Filtered for Positives)
logging.info("Loading Philosophical STS dataset...")
philosophy_ds = load_dataset("johnnyboycurtis/Philosophical-STS-Text-Pairs", split="train")

# Filter: Keep only rows where score > 0.7
logging.info(f"Original Philosophy size: {len(philosophy_ds)}")
philosophy_ds = philosophy_ds.filter(lambda x: x['llm_score'] > 0.7)
logging.info(f"Filtered Philosophy size (>0.7): {len(philosophy_ds)}")

# Map columns: (txt1, txt2) -> (anchor, positive)
# We drop 'score' because for MultipleNegativesRankingLoss, we just need to know they are positive pairs.
philosophy_ds = philosophy_ds.map(
    lambda x: {'anchor': x['text1'], 'positive': x['text2']},
    remove_columns=['text1', 'text2', 'llm_score']
)


# Combine for Training
train_dataset = {
    "nli": nli_dataset,
    "quora": quora_dataset,
    "natural_questions": natural_questions,
    "msmarco": msmarco_dataset,
    "gooaq": gooaq_dataset,
    "philosophy": philosophy_ds, 
}
print(train_dataset)

logging.info(f"Training with {len(train_dataset)} datasets: {list(train_dataset.keys())}")

# --- 3. Define Loss Functions ---
logging.info("\nDefining Matryoshka-wrapped Loss Functions...")

mrl_dims = [384, 192, 64]

# A. Symmetric Loss (NLI, Quora)
# CachedMNRL works best with batch size 64-128
inner_loss_symmetric = losses.CachedMultipleNegativesSymmetricRankingLoss(model, mini_batch_size=64)
loss_symmetric = losses.MatryoshkaLoss(model, inner_loss_symmetric, matryoshka_dims=mrl_dims)

# B. Asymmetric Loss (MSMARCO, NQ, GooAQ)
loss_asymmetric = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=64)

loss_functions = {
    "nli": loss_symmetric,
    "quora": loss_symmetric,
    "natural_questions": loss_asymmetric,
    "msmarco": loss_asymmetric,
    "gooaq": loss_asymmetric,
    "philosophy": loss_symmetric,
}

# --- 4. Training Arguments ---
logging.info("Configuring training arguments...")

args = SentenceTransformerTrainingArguments(
    output_dir=str(output_dir),
    
    # --- OPTIMIZATION FOR BLANK MODEL ---
    num_train_epochs=2,              # 2 Epochs is plenty given dataset size (Millions of rows)
    learning_rate=3e-4,              # INCREASED: 1e-4 for training from scratch
    per_device_train_batch_size=256, # High batch size = better negatives
    warmup_ratio=0.1,                # 10% warmup to stabilize initial random gradients
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    
    # Hardware
    fp16=False,
    bf16=True, 
    bf16_full_eval=True,
    
    # Batch Strategy
    batch_sampler=BatchSamplers.NO_DUPLICATES, 
    multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL,

    # Evaluation & Saving
    eval_strategy="steps",
    eval_steps=2000,                 # Increased steps because dataset is huge
    save_strategy="steps",
    save_steps=2000,
    save_total_limit=4,
    
    # Tracking
    load_best_model_at_end=True,
    metric_for_best_model="eval_sts-dev_spearman_cosine", # Explicitly tracking NLI accuracy
    logging_steps=100,
    report_to="none" 
)

# --- 5. Evaluators ---
logging.info("\nSetting up evaluators...")

# A. NLI (Accuracy Check)
eval_nli = load_dataset("sentence-transformers/all-nli", "triplet", split="dev")
nli_evaluator = TripletEvaluator(
    anchors=list(eval_nli["anchor"]),     
    positives=list(eval_nli["positive"]), 
    negatives=list(eval_nli["negative"]), 
    name="all-nli-dev",
)

# B. STS (Correlation Check)
stsb_eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")
dev_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=list(stsb_eval_dataset["sentence1"]), # Add list()
    sentences2=list(stsb_eval_dataset["sentence2"]), # Add list()
    scores=list(stsb_eval_dataset["score"]),         # Add list()
    main_similarity=SimilarityFunction.COSINE,
    name="sts-dev",
)

# Combine
evaluator = SequentialEvaluator([nli_evaluator, dev_evaluator])

# --- 6. Train ---
logging.info("\nInitializing Trainer...")
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=loss_functions,
    evaluator=evaluator,
)

logging.info("🚀🚀🚀 STARTING Pre-Training 🚀🚀🚀")
trainer.train()
logging.info("🏁🏁🏁 Pre-Training COMPLETE 🏁🏁🏁")

# --- 7. Save ---
final_model_save_path = output_dir / "final_model"
model.save(str(final_model_save_path))
logging.info(f"\n✅ Final model saved to: {final_model_save_path}")

# --- 8. Final Eval ---
logging.info("Running final evaluation (Re-using Sequential Evaluator)...")
# Removed 'msmarco_evaluator' because it wasn't defined.
# We reuse the robust Sequential evaluator we defined above.
evaluator(model)
