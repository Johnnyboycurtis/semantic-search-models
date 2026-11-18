# ModernBERT-small-v2 Training Roadmap

This guide lays out the end-to-end training pipeline for `ModernBERT-small-v2` (twelve layers, three hundred eighty-four dimensions, six attention heads) on a single NVIDIA A100 GPU, using `Alibaba-NLP/gte-modernbert-base` as the teacher.

Every stage below has one specific job. Each stage builds on the previous checkpoint without throwing away earlier work.

---

## Stage 1: GUIDE Initialization (Already Completed / Updated)

- **Script:** `ModernBERT-small-v2/mlm/init_model_mlm.py`
- **What It Does:** Compresses the teacher's seven-hundred-sixty-eight-dimension token embedding table down to three hundred eighty-four dimensions using SVD, and projects **only Layer zero** into the student. Layers one through eleven start randomly initialized.
- **Why It Matters:** Initializing more than Layer zero traps deeper layers in rigid teacher subspaces and hurts final quality. Starting with SVD embeddings plus Layer zero gives the student immediate syntax ability while leaving Layers one through eleven free to organize around three hundred eighty-four dimensions.

---

## Stage 2: MLM Warmup (Your Current Run — Ready to Stop)

- **Scripts:** `ModernBERT-small-v2/mlm/pre_train.py` and `ModernBERT-small-v2/mlm/pre_train_phase1b_optional.py`
- **What It Does:** Trains the student on masked language modeling. Your Phase 1B run already unfroze `tok_embeddings` and adapted global RoPE theta (`160000.0`) to four thousand ninety-six tokens, reaching a validation loss plateau around `1.833` at `checkpoint-10000`.
- **Why We Stop Here:** MLM teaches token reconstruction (guessing masked words), not semantic search. Once the random layers have stabilized and validation loss plateaus, more MLM gives diminishing returns. Note: if you ever restart from scratch, you can skip MLM completely and go straight from Stage 1 into Stage 3.

---

## Stage 3: MiniLMv2 Relation Distillation (Next Step)

- **Script:** `ModernBERT-small-v2/minilmv2_distill.py`
- **Input Checkpoint:** Your Phase 1B checkpoint (`checkpoint-10000`), with **all student weights unfrozen**.
- **Output Checkpoint:** `./minilmv2-student/final`
- **What It Teaches the Model (Internal Engine):** Instead of looking at final sentence vectors, MiniLMv2 works inside the Transformer on raw, unlabeled text (five-hundred-twelve-token packed chunks). It teaches your student's final layer (layer eleven) to mimic the Query-Query, Key-Key, and Value-Value attention relations of teacher layer seventeen using forty-eight relation heads.
- **Why It Matters:** It transfers *how the teacher reads and connects tokens* directly into your six-head student without needing matching head counts.

---

## Stage 4: Dense Retrieval Distillation

- **Script:** `ModernBERT-small-v2/distil_train.py`
- **Input Checkpoint:** `./minilmv2-student/final` loaded as a `SentenceTransformer` (mean pooling plus L2 normalization, all weights unfrozen).
- **Output Checkpoint:** `ModernBERT-small-distilled-v2/final`
- **What It Teaches the Model (Output Search Space):** Aligns the student's final three-hundred-eighty-four-dimension sentence vectors to `gte-modernbert-base`'s embedding geometry using **Cosine / InfoNCE (with in-batch negatives) plus MSE**, evaluated on NanoBEIR (`MSMARCO` and `HotpotQA`).
- **Why It Matters:** Stage 3 aligned the student's internal attention; Stage 4 places the final pooled vectors into the right neighborhood for semantic search.

---

## Stage 5: Supervised Hard-Negative Fine-Tuning

- **Input Checkpoint:** `ModernBERT-small-distilled-v2/final`
- **What It Teaches the Model (Precision Ranking):** Following the Nomic Embed recipe, train for **one epoch** on curated query-passage pairs using `search_query` and `search_document` prefixes, InfoNCE loss, and seven randomly sampled hard negatives per query.
- **Why It Matters:** Teaches the model to separate true answers from "almost-right" distractors that share keywords with the query.

---

## Consolidated Engineering Principles (From Earlier Notes)

- **Why SVD Over PCA for Embeddings:** PCA subtracts the mean vector before decomposing, which shifts the origin of the embedding space and distorts dot products and norms. SVD operates directly on the teacher embedding table without shifting the origin, preserving pairwise dot products.
- **Why All Student Weights Stay Unfrozen:** GUIDE is a warm start, not a lock. Because compressing seven hundred sixty-eight dimensions down to three hundred eighty-four is lossy, keeping `tok_embeddings` and Layer zero trainable lets the entire network co-adapt.
- **Single-A100 VRAM & Optimizer Playbook:** Always use `bfloat16` mixed precision with `sdpa` attention and `gradient_checkpointing` (`use_reentrant=False`). Use `adamw_torch_fused` when VRAM allows, or `adamw_bnb_8bit` if you need to save optimizer memory. For dense distillation (Stage 4), pre-compute teacher embeddings to disk so the teacher does not occupy VRAM during training.

