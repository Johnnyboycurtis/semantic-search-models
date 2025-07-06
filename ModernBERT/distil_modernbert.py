# ==============================================================================
#                  Improving a Model with Knowledge Distillation (Advanced)
# ==============================================================================
#
# PURPOSE:
# This script uses an advanced knowledge distillation workflow based on the
# `SentenceTransformerTrainer`. It improves a small "student" model by training
# it to mimic a powerful "teacher" model.
#
# KEY FEATURES OF THIS SCRIPT:
# 1.  **Trainer-based:** Uses the modern `SentenceTransformerTrainer` for robust training.
# 2.  **PCA Projection:** Automatically handles dimension differences between the
#     teacher and student by learning a PCA projection layer.
# 3.  **Pre-computation:** Efficiently pre-computes teacher embeddings for faster training.
# 4.  **Comprehensive Evaluation:** Tracks both STSb performance and MSE mimicry loss.
#
# ==============================================================================

import logging
from datetime import datetime
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from sklearn.decomposition import PCA
import pandas as pd

from sentence_transformers import (
    LoggingHandler,
    SentenceTransformer,
    models,
    losses,
    evaluation,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

# --- Configuration ---
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

start = datetime.now()
logging.info(f"Start time: {start}")

# --- Step 1: Define Teacher and Student Models ---
# The "teacher" model is a powerful, pre-trained model. `bge-base` is a top-tier choice.
teacher_model_name = "sentence-transformers/all-MiniLM-L6-v2"
# teacher_model_name = "BAAI/bge-base-en-v1.5"
# teacher_model_name = "Alibaba-NLP/gte-modernbert-base" # too large!!
# The "student" model is our own small ModernBERT.
student_model_path = './ModernBERT-small/training-small-modernbert/final'

# Define where we will save the final, distilled model.
output_dir = "ModernBERT-small/distilled-modernbert-small" # + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Training hyperparameters
train_batch_size = 64
inference_batch_size = 64 # For the teacher model's encoding step

logging.info(f"Teacher model: {teacher_model_name}")
logging.info(f"Student model (initial): {student_model_path}")

teacher_model = SentenceTransformer(teacher_model_name, model_kwargs={"torch_dtype": torch.bfloat16})
print(teacher_model)
student_model = SentenceTransformer(student_model_path,  model_kwargs={"torch_dtype": torch.bfloat16})
print(student_model)


# --- Step 2: Load and Prepare Datasets ---
# We use a diverse set of sentences for distillation.
# The combination of NLI and Wikipedia provides a good mix of formal and informal text.
logging.info("Loading NLI and Wikipedia datasets...")
nli_dataset = load_dataset("sentence-transformers/all-nli", "pair-score", split="train")
# To make the dataset more diverse, we'll just use the sentences, not the pairs.
nli_dataset = nli_dataset.map(
    lambda batch: {"sentence": batch["sentence1"] + batch["sentence2"]},
    batched=True,
    remove_columns=nli_dataset.column_names,
)
# Remove duplicates
nli_dataset = Dataset.from_pandas(pd.DataFrame(nli_dataset).drop_duplicates(), preserve_index=False)

wiki_dataset = load_dataset("sentence-transformers/wikipedia-en-sentences", split="train")

# For demonstration, we'll use a smaller subset. For best results, use the full datasets.
train_dataset = concatenate_datasets([
    nli_dataset.select(range(400000)),
    wiki_dataset.select(range(400000))
])
logging.info(f"Combined training dataset size: {len(train_dataset):,}")


# --- Step 3: Handle Mismatched Dimensions with PCA ---
# Our student (256 dims) is smaller than the teacher (768 dims). We need to
# project the teacher's embeddings down to the student's size.
student_embedding_dim = student_model.get_sentence_embedding_dimension()
teacher_embedding_dim = teacher_model.get_sentence_embedding_dimension()
print("Embedding Dimensions: ", student_embedding_dim, teacher_embedding_dim)

if student_embedding_dim < teacher_embedding_dim:
    logging.info("Student dimension < Teacher dimension. Applying PCA projection.")

    # We train PCA on a sample of the teacher's embeddings
    pca_train_sentences = list(nli_dataset.select(range(30000))['sentence'])
    pca_embeddings = teacher_model.encode(pca_train_sentences, convert_to_numpy=True)

    pca = PCA(n_components=student_embedding_dim)
    pca.fit(pca_embeddings)

    # We add a new Dense layer to the teacher model that applies the PCA projection
    dense_layer = models.Dense(
        in_features=teacher_embedding_dim,
        out_features=student_embedding_dim,
        bias=False,
        activation_function=torch.nn.Identity(),
    )
    dense_layer.linear.weight = torch.nn.Parameter(torch.tensor(pca.components_, dtype=torch.bfloat16))
    teacher_model.add_module("dense_pca", dense_layer)
    logging.info(f"Teacher model updated. Output dimension is now: {teacher_model.get_sentence_embedding_dimension()}")


# --- Step 4: Pre-compute Teacher Embeddings ---
# This is the most time-consuming step. We run the entire training dataset
# through the (potentially updated) teacher model to get the target "label" embeddings.
logging.info("Mapping dataset with teacher embeddings... (This may take a while)")
def map_embeddings(batch):
    return {"label": teacher_model.encode(batch["sentence"], batch_size=inference_batch_size).tolist()}

train_dataset = train_dataset.map(map_embeddings, batched=True, batch_size=2048)
# Optionally save to disk to skip this step in future runs
# train_dataset.save_to_disk("datasets/distillation_train_dataset_modernbert")


# --- Step 5: Set up Evaluators and Loss ---
# We need an evaluation set to monitor progress. We'll use STSb.
stsb_eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")

# Evaluator 1: Standard STSb evaluation to check real-world performance.
dev_evaluator_stsb = EmbeddingSimilarityEvaluator(
    sentences1=stsb_eval_dataset["sentence1"],
    sentences2=stsb_eval_dataset["sentence2"],
    scores=stsb_eval_dataset["score"],
    name="sts-dev"
)

# Evaluator 2: MSE evaluator to check how well the student mimics the teacher.
# For this, we need a small sample of sentences.
eval_sentences = stsb_eval_dataset['sentence1'][:2000] + stsb_eval_dataset['sentence2'][:2000]
dev_evaluator_mse = evaluation.MSEEvaluator(eval_sentences, eval_sentences, teacher_model=teacher_model, name="mse-dev")

# Combine them to run both during evaluation steps.
evaluator = evaluation.SequentialEvaluator([dev_evaluator_stsb, dev_evaluator_mse])

# The loss function remains Mean Squared Error.
train_loss = losses.MSELoss(model=student_model)


# --- Step 6: Configure and Run the Trainer ---
args = SentenceTransformerTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=2,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=train_batch_size,
    warmup_ratio=0.1,
    fp16=False,
    bf16=True,
    bf16_full_eval=True,
    learning_rate=1e-4, # Distillation often benefits from a slightly higher learning rate
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=2,
    logging_steps=100,
    metric_for_best_model="sts-dev_spearman_cosine", # We care most about the real-world performance
    load_best_model_at_end=True,
    run_name="distilled-modernbert-small",
)

trainer = SentenceTransformerTrainer(
    model=student_model,
    args=args,
    train_dataset=train_dataset,
    evaluator=evaluator,
    loss=train_loss,
)

logging.info("🚀🚀🚀 STARTING KNOWLEDGE DISTILLATION 🚀🚀🚀")
trainer.train()
logging.info("🏁🏁🏁 DISTILLATION COMPLETE 🏁🏁")

# Save the final, best-performing model
final_output_dir = f"{output_dir}/final"
student_model.save(final_output_dir)
logging.info(f"Final distilled model saved to: {final_output_dir}")

end = datetime.now()
logging.info(f"Start time: {start} -- End time: {end}")
logging.info(f"Duration: {end-start}")