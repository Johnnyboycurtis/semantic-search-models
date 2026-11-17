# Pre-Training & Distillation Strategy: `ModernBERT-small-v2` (Teaching Guide)

**Target Model:** 12-layer, 384-dim ("Deep & Narrow") ModernBERT embedding model (6 attention heads, 64 `head_dim`, 768 GeGLU intermediate size, 50,368 vocab, up to 1024+ context).  
**Teacher Model:** `Alibaba-NLP/gte-modernbert-base` (22 layers, 768-dim, 12 attention heads).  
**Hardware Constraint:** Single NVIDIA A100 GPU (limited VRAM and time budget).

---

## 1. Current Process vs. Recommended Process

| Stage | Current Process | Recommended Process | Why Change? |
| :--- | :--- | :--- | :--- |
| **1. Init** | SVD `tok_embeddings` + slice **all 12 layers** | **GUIDE:** PCA/SVD `tok_embeddings` + **Layer 0 only** (Layers 1–11 random) | GUIDE ablations show initializing >1 layer degrades performance by trapping deeper layers in sliced subspaces. |
| **2. Warmup / Pre-train** | Two-stage MLM (1,024 → 4,096 tokens) with `tok_embeddings` **frozen** | **Short MLM warmup** (`512` tokens) + **MiniLMv2 Relation Distillation**, all student weights **unfrozen** | Pure MLM teaches token reconstruction, not retrieval. Unfreezing lets the 384-d embeddings adapt; MiniLMv2 transfers attention structure fast. |
| **3. Dense Distill** | `MSELoss` on PCA-reduced teacher embeddings | **Cosine / InfoNCE (in-batch negatives) + MSE** using pre-computed teacher embeddings | Pure MSE only supervises Euclidean distance to a compressed point; Cosine/InfoNCE teaches ranking geometry. |
| **4. Fine-tune** | Planned (TBD) | Hard negatives / triplets (and optional long-context pass) | Specializes the model for your target retrieval domain. |

---

## 2. Core Concepts Explained

### A. Why GUIDE Initializes Only Embeddings + Layer 0
- **The Intuition:** Layer 0 is responsible for "cleaning up" raw token embeddings and learning basic syntax. By compressing the teacher's embedding table ($768 \to 384$) via PCA/SVD and projecting Layer 0 to match, the student skips the "infancy" phase of learning basic word relationships.
- **Why Stop at Layer 0?** Deep layers (Layers 1–11) need freedom to organize themselves around the narrower 384-dim width. Slicing all 12 layers forces them into rigid, suboptimal subspaces.
- **Why SVD over PCA?**
  - **No mean-centering distortion:** PCA subtracts the mean vector before decomposing, which shifts the origin of the embedding space. In Transformers, dot-product attention and LayerNorm/RMSNorm depend on the original origin and vector norms—SVD preserves the raw geometry and pairwise dot products ($E_S E_S^T \approx E_T E_T^T$) without shifting the space.
  - **Numerical stability & memory:** SVD operates directly on the embedding matrix $E_T$ ($50,368 \times 768$) without materializing a $768 \times 768$ covariance matrix, and naturally supports **frequency-weighted SVD** (weighting common tokens higher than rare tokens).

### B. Freezing vs. Unfreezing: Teacher vs. Student
- **Rule of Thumb:** **Teacher = Frozen. Student = Completely Unfrozen.**
- **Why unfreezing `tok_embeddings` and Layer 0 does NOT hurt distillation:**
  - GUIDE is a **warm start** (a good starting point), not a permanent constraint. Gradient descent will smoothly adapt those weights toward the distillation target rather than erasing them.
  - **What goes wrong if you freeze `tok_embeddings`:** PCA compression ($768 \to 384$) is lossy. Freezing `tok_embeddings` forces Layers 1–11 to compensate for a rigid input space, and prevents token representations from shifting to group semantically similar query/document terms together.

### C. MiniLMv2: The Method vs. The Model
- **The Model:** `all-MiniLM-L6-v2` is a famous sentence-embedding model.
- **The Method (what we use here):** *MiniLMv2* (Wang et al., 2021) is a **task-agnostic distillation technique**:
  1. Take `Q`, `K`, and `V` vectors from a single teacher layer and the student's **last layer** (layer 11).
  2. Concatenate across attention heads and re-split into $A_r$ **relation heads** (e.g., `48` for base teachers). This **decouples head counts**, so your **6-head student** can learn directly from a **12-head teacher**.
  3. Minimize KL-divergence on **`Q-Q`, `K-K`, and `V-V`** scaled dot-product relations.
  4. **Layer Selection:** For deep teachers (like 22-layer `gte-modernbert-base`), distill from an **upper-middle layer** (~75–85% depth, e.g., layer 17–19) rather than the final layer, which is often overly specialized.

### D. Why Raw MSE Alone Is Limiting in Dense Distillation
- Regressing only the final 384-d pooled vector with `MSELoss` forces 12 layers to guess how to route information to match a single point, and PCA-projecting the teacher's output can distort ranking geometry.
- **Better approach:** Combine **Cosine Similarity / InfoNCE (with in-batch negatives)** + a small **MSE** term. This directly optimizes *relative ranking* (what semantic search actually measures).

---

## 3. Single-A100 VRAM Playbook

If you are hitting VRAM limits on a single A100, use these levers:
1. **Cap Sequence Length at `512` During Teacher-Student Training:** Attention memory scales quadratically with sequence length. Do not run teacher + student together at `4,096` tokens on a single A100.
2. **Mixed Precision + Flash Attention 2:** Always set `bf16=True` and `attn_implementation="flash_attention_2"` (avoids materializing the $L \times L$ attention matrix).
3. **Gradient Checkpointing:** Enable `gradient_checkpointing=True` on the student (~60–70% activation memory savings for ~20% extra compute).
4. **8-Bit Optimizer:** Use `optim="adamw_bnb_8bit"` (`bitsandbytes`) to cut optimizer state VRAM by ~75%.
5. **Micro-Batch + Gradient Accumulation:** Lower `per_device_train_batch_size` (e.g., `16` or `32`) and raise `gradient_accumulation_steps` to keep effective batch size high.
6. **Pre-Compute Teacher Embeddings (Stage 3):** Run `gte-modernbert-base` offline to save target embeddings to disk so the teacher doesn't sit in VRAM during dense distillation.

---

## 4. Recommended Step-by-Step Pipeline

1. **Baseline Diagnostic (Optional, Zero-Waste):** Run Phase 2 (`SentenceTransformerTrainer` with `tok_embeddings` unfrozen) on your existing Phase 1 checkpoint to get a NanoBEIR baseline.
2. **Stage 1 — GUIDE Init:** PCA/SVD `tok_embeddings` ($768 \to 384$) + Layer 0 projection; leave Layers 1–11 random.
3. **Stage 2 — Short MLM Warmup (`512` tokens, `bf16`, unfrozen):** Brief run to wake up Layers 1–11.
4. **Stage 3 — MiniLMv2 Relation Distillation (`512` tokens, `bf16`, unfrozen):** Distill `Q-Q + K-K + V-V` ($A_r = 48$) from `gte-modernbert-base` (layer ~17–19) into student layer 11.
5. **Stage 4 — Dense Retrieval Distillation:** Train on pre-computed `gte-modernbert-base` embeddings using **Cosine / InfoNCE + MSE**, evaluated on NanoBEIR.
6. **Stage 5 — Downstream Fine-Tuning:** Fine-tune on hard negatives / triplets (and optionally a brief long-context pass without the teacher).
