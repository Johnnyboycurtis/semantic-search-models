import logging
import torch
import pandas as pd
from pathlib import Path
from datasets import load_dataset, Dataset
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer, models

# --- Configuration ---
TEACHER_MODEL_NAME = "Alibaba-NLP/gte-modernbert-base"
# The dimension your student model uses (ModernBERT-small is likely 384 or 768)
# Ensure this matches the student model you plan to train.
STUDENT_DIMENSION = 384 
RANDOM_SEED = 384

INPUT_PARQUET_PATH = "mlm/data/combined_mlm_dataset.parquet"
OUTPUT_DATASET_PATH = "datasets/distillation_train_dataset"

INFERENCE_BATCH_SIZE = 128
PCA_SAMPLE_SIZE = 2000 # Number of sentences to use to calculate PCA projection
DTYPE = torch.bfloat16

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)

# 1. Load Teacher Model
logging.info(f"Loading Teacher Model: {TEACHER_MODEL_NAME}")
teacher_model = SentenceTransformer(
    TEACHER_MODEL_NAME,
    model_kwargs={
        "attn_implementation": "sdpa",
        "dtype": torch.bfloat16, # Speed up encoding
        "device_map": "cuda",
    },
)

# 2. Load Local Parquet Dataset
logging.info(f"Loading local dataset from: {INPUT_PARQUET_PATH}")
# We load as a HuggingFace dataset directly from parquet
dataset = load_dataset("parquet", data_files=INPUT_PARQUET_PATH, split="train")

# Ensure the column is named 'text' for consistency
if "text" not in dataset.column_names:
    raise ValueError(f"Missing `text` columnin {dataset.column_names}")

logging.info(f"Dataset loaded. Total rows: {len(dataset)}")

# 3. Handle Dimensionality Reduction (PCA)
# If the teacher (768) is larger than the student (e.g. 384), we must project now.
teacher_dim = teacher_model.get_sentence_embedding_dimension()
if STUDENT_DIMENSION < teacher_dim:
    logging.info(f"Teacher dim ({teacher_dim}) > Student dim ({STUDENT_DIMENSION}). Computing PCA...")
    
    # Take a sample for PCA fitting
    pca_sample = dataset.shuffle(seed=RANDOM_SEED).select(range(min(len(dataset), PCA_SAMPLE_SIZE)))
    pca_embeddings = teacher_model.encode(
        pca_sample["text"], 
        convert_to_numpy=True, 
        show_progress_bar=True
    )
    
    pca = PCA(n_components=STUDENT_DIMENSION)
    pca.fit(pca_embeddings)

    # Create a Dense layer and add it to the teacher
    dense = models.Dense(
        in_features=teacher_dim,
        out_features=STUDENT_DIMENSION,
        bias=False,
        activation_function=torch.nn.Identity(),
    )
    # Convert the PCA components to the same dtype as the teacher model (BFloat16)
    pca_weights = torch.tensor(pca.components_, dtype=torch.bfloat16)
    dense.linear.weight = torch.nn.Parameter(pca_weights)
    teacher_model.add_module("dense", dense)
    logging.info(f"PCA projection layer added to teacher. New output dim: {STUDENT_DIMENSION}")
else:
    logging.info(f"No dimension reduction needed (Teacher: {teacher_dim}, Student: {STUDENT_DIMENSION})")

# 4. Map Embeddings (Generate Teacher Labels)
logging.info("Starting embedding generation...")

def map_embeddings(batch):
    # We use the teacher to encode the sentences
    # The output 'label' is what the student will try to mimic via MSE loss
    embeddings = teacher_model.encode(
        batch["text"], 
        batch_size=INFERENCE_BATCH_SIZE, 
        show_progress_bar=False,
        convert_to_tensor=True
    )
    return {"label": embeddings.half().cpu().numpy()}

# Using map with a large batch size to minimize overhead
distill_dataset = dataset.map(
    map_embeddings, 
    batched=True, 
    batch_size=500, # Number of rows passed to map_embeddings at once
)

# 5. Save to Disk
logging.info(f"Saving pre-computed dataset to: {OUTPUT_DATASET_PATH}")
distill_dataset.save_to_disk(OUTPUT_DATASET_PATH)

# Optional: Save the modified teacher model if PCA was used 
# so you can use the same projection during evaluation scripts.
if STUDENT_DIMENSION < teacher_dim:
    teacher_save_path = Path("teacher_model_reduced")
    teacher_model.save(str(teacher_save_path))
    logging.info(f"Reduced teacher model saved to {teacher_save_path}")

logging.info("Data preparation complete!")

