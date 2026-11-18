"""
Stage 4: Dense Retrieval Distillation.

Aligns the MiniLMv2-distilled student's final 384-dim sentence embeddings to
pre-computed teacher embeddings from teacher_model_reduced using a combined
MSE + Cosine Distance loss, evaluated on NanoBEIR (MSMARCO, HotpotQA) and MSE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer, evaluation
from sentence_transformers.evaluation import NanoBEIREvaluator
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

# --- Configuration ---
TEACHER_MODEL_PATH = "teacher_model_reduced"
STUDENT_MODEL_PATH = "/content/drive/MyDrive/data/models/minilmv2-student/final"
DATASET_DIR = "datasets/distillation_train_dataset"
OUTPUT_DIR = "ModernBERT-small-distilled-v2"

BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 2
COSINE_WEIGHT = 0.5


class CosinePlusMSELoss(nn.Module):
    """Combines MSE and cosine distance against pre-computed teacher embeddings."""

    def __init__(self, model: SentenceTransformer, cosine_weight: float = 0.5):
        super().__init__()
        self.model = model
        self.cosine_weight = cosine_weight

    def forward(self, sentence_features, labels):
        # Shape: batch size, student hidden dimension (384)
        student_embs = self.model(sentence_features[0])["sentence_embedding"]
        target_embs = labels.to(device=student_embs.device, dtype=student_embs.dtype)

        mse = F.mse_loss(student_embs, target_embs)
        cosine_dist = (1.0 - F.cosine_similarity(student_embs, target_embs, dim=-1)).mean()
        return mse + self.cosine_weight * cosine_dist


def main():
    # 1. Load Teacher Model (used only by MSEEvaluator during validation)
    print(f"Loading Teacher Model from: {TEACHER_MODEL_PATH}")
    teacher_model = SentenceTransformer(
        TEACHER_MODEL_PATH,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
            "device_map": "cuda",
        },
    )

    # 2. Load Student Model (all weights unfrozen)
    print(f"Loading Student Model from: {STUDENT_MODEL_PATH}")
    student_model = SentenceTransformer(
        STUDENT_MODEL_PATH,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
            "device_map": "cuda",
        },
    )
    for param in student_model.parameters():
        param.requires_grad = True

    # 3. Load Pre-computed Dataset
    print(f"Loading pre-computed dataset from: {DATASET_DIR}")
    train_dataset = load_from_disk(DATASET_DIR)

    if "text" in train_dataset.column_names:
        train_dataset = train_dataset.rename_column("text", "sentence")

    print(f"Dataset Columns: {train_dataset.column_names}")

    # 4. Prepare Evaluators
    print("Preparing Evaluators...")
    retrieval_evaluator = NanoBEIREvaluator(
        dataset_names=["MSMARCO", "HotpotQA"],
        batch_size=BATCH_SIZE,
    )

    eval_sample = train_dataset.select(range(min(len(train_dataset), 2000)))
    eval_sentences = eval_sample["sentence"]
    dev_evaluator_mse = evaluation.MSEEvaluator(
        source_sentences=eval_sentences,
        target_sentences=eval_sentences,
        teacher_model=teacher_model,
        name="mse-dev",
    )

    dev_evaluator = evaluation.SequentialEvaluator(
        [dev_evaluator_mse, retrieval_evaluator]
    )

    # 5. Define Loss & Training Arguments (A100 bfloat16)
    train_loss = CosinePlusMSELoss(model=student_model, cosine_weight=COSINE_WEIGHT)

    args = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_steps=0.1,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        adam_beta2=0.98,
        optim="adamw_torch_fused",
        bf16=True,
        fp16=False,
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,
        logging_steps=100,
        run_name="modernbert-distillation",
        load_best_model_at_end=True,
        metric_for_best_model="eval_NanoMSMARCO_cosine_ndcg@10",
    )

    # 6. Start Training
    trainer = SentenceTransformerTrainer(
        model=student_model,
        args=args,
        train_dataset=train_dataset,
        loss=train_loss,
        evaluator=dev_evaluator,
    )

    print("Starting dense retrieval distillation...")
    trainer.train()

    # 7. Final Save
    print(f"Saving final model to {OUTPUT_DIR}/final")
    student_model.save(f"{OUTPUT_DIR}/final")


if __name__ == "__main__":
    main()
