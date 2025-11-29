"""
ModernBERT-Embed-Small Training Script
--------------------------------------
A "Home Lab SOTA" recipe combining strategies from:
1. GUIDE (Google): Uniform Layer Pruning for Initialization.
2. Gecko (DeepMind): Distilling from Cross-Encoder scores (Offline).
3. M3-Embedding (BAAI): Efficient batching and Matryoshka representation.
4. IBM Granite: Using MarginMSE for dense signal distillation.

Author: JohnnyBoyCurtis (assisted by AI)
Model Name: ModernBERT-Embed-Small
"""

import logging
import sys
import torch
from pathlib import Path
from transformers import AutoModel, AutoConfig
from datasets import load_from_disk
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.losses import MarginMSELoss, MatryoshkaLoss

# --- Configuration ---
TEACHER_MODEL_ID = "answerdotai/ModernBERT-base"
OUTPUT_DIR = "ModernBERT-Embed-Small"
DATASET_PATH = "./scored_training_data" # Path to your Gecko-scored dataset

# Hyperparameters (Optimized for Consumer GPUs)
BATCH_SIZE = 64        # MarginMSE is efficient; 64 is sufficient
NUM_EPOCHS = 3         # Distillation converges fast (weights are already smart)
LEARNING_RATE = 2e-5   # Low LR because we preserve GUIDE initialization
MAX_SEQ_LENGTH = 512   # Standard for dense retrieval
MATRYOSHKA_DIMS = [768, 512, 256, 128, 64] # Enable flexible vector sizes

# Setup Logging
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# PHASE 1: GUIDE INITIALIZATION (The "Smart Surgery")
# Reference: GUIDE: Guided Initialization and Distillation of Embeddings (2025)
# ==============================================================================
def initialize_via_guide(teacher_id, save_path):
    """
    Creates a 6-layer student by copying uniform layers [0, 4, 8, 12, 16, 20]
    from the 22-layer teacher. This reduces teacher-student gap by ~26% pre-training.
    """
    logger.info(f"🎨 Applying GUIDE: Initializing Student from {teacher_id}...")
    
    # 1. Load Teacher (CPU is fine for this operation)
    teacher_config = AutoConfig.from_pretrained(teacher_id)
    teacher = AutoModel.from_pretrained(teacher_id, config=teacher_config)
    
    # 2. Configure Student (Same architecture, fewer layers)
    student_config = AutoConfig.from_pretrained(teacher_id)
    student_config.num_hidden_layers = 6
    student = AutoModel.from_config(student_config)
    
    # 3. Perform Weight Surgery (Uniform Selection)
    # ModernBERT-base has 22 layers. We select 6 evenly spaced.
    # Indices: 0 (Start), 4, 8, 12, 16, 20 (End) - Captures full semantic depth
    teacher_indices = [0, 4, 8, 12, 16, 20]
    
    # Copy Embeddings (The Foundation)
    student.embeddings.load_state_dict(teacher.embeddings.state_dict())
    
    # Copy Encoder Layers
    for student_idx, teacher_idx in enumerate(teacher_indices):
        logger.info(f"   -> Transplanting Teacher Layer {teacher_idx} to Student Layer {student_idx}")
        student.encoder.layers[student_idx].load_state_dict(
            teacher.encoder.layers[teacher_idx].state_dict()
        )
        
    # Copy Final Norm / Heads
    student.final_norm.load_state_dict(teacher.final_norm.state_dict())
    
    # 4. Save the "Smart" Initialization
    logger.info(f"💾 Saving GUIDE-initialized model to {save_path}")
    student.save_pretrained(save_path)
    
    # Cleanup memory
    del teacher
    del student
    torch.cuda.empty_cache()

# ==============================================================================
# PHASE 2: TRAINING (Distillation)
# Reference: IBM Granite / Gecko / M3-Embedding
# ==============================================================================

def main():
    # 1. Check if we need to run GUIDE init
    init_path = Path(OUTPUT_DIR) / "guide_init"
    if not init_path.exists():
        initialize_via_guide(TEACHER_MODEL_ID, str(init_path))
    
    # 2. Load the Student Model
    logger.info("🚀 Loading Student Model...")
    model = SentenceTransformer(
        str(init_path),
        model_card_data=SentenceTransformerModelCardData(
            model_name="ModernBERT-Embed-Small",
            language="en",
            license="apache-2.0",
        )
    )
    model.max_seq_length = MAX_SEQ_LENGTH

    # 3. Load Scored Data (The Gecko Strategy)
    # We expect the dataset to have a 'label' column which is the score margin
    # margin = Teacher_Score(Anchor, Positive) - Teacher_Score(Anchor, Negative)
    logger.info(f"📂 Loading Dataset from {DATASET_PATH}...")
    train_dataset = load_from_disk(DATASET_PATH)
    
    # Ensure columns match MarginMSE expectation
    # If your offline script named it 'margin_score', rename it to 'label' here
    if "margin_score" in train_dataset.column_names:
        train_dataset = train_dataset.rename_column("margin_score", "label")

    # 4. Define Loss Function (IBM Strategy)
    # MarginMSELoss: Teaches the model "How much better" A is than B.
    # MatryoshkaLoss: Forces the model to pack information into front dimensions.
    logger.info("⚖️  Setting up MarginMSE + Matryoshka Loss...")
    base_loss = MarginMSELoss(model)
    
    train_loss = MatryoshkaLoss(
        model=model,
        loss=base_loss,
        matryoshka_dims=MATRYOSHKA_DIMS
    )

    # 5. Training Arguments (M3 Strategy)
    # group_by_length=True is crucial for efficiency (minimizes padding)
    args = SentenceTransformerTrainingArguments(
        output_dir=str(Path(OUTPUT_DIR) / "checkpoints"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.1,
        fp16=True,             # Use bf16=True if on Ampere (3090/4090/A100)
        group_by_length=True,  # M3-Embedding efficiency trick
        logging_steps=50,
        save_strategy="epoch",
        report_to="none",      # Set to "wandb" if you want tracking
    )

    # 6. Initialize Trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=train_loss,
    )

    # 7. Train
    logger.info("🔥 Starting Distillation Training...")
    trainer.train()

    # 8. Save Final Model
    final_path = Path(OUTPUT_DIR) / "final"
    logger.info(f"✅ Training Complete. Saving to {final_path}")
    model.save_pretrained(str(final_path))
    
    # Optional: Push to Hub
    # model.push_to_hub("johnnyboycurtis/ModernBERT-Embed-Small")

if __name__ == "__main__":
    main()
