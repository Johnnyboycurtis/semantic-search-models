# Pre-Training & Distillation Strategy: `ModernBERT-small-v2` (Lead ML Engineer's Teaching Guide)

**Target Architecture:** 12-layer, 384-dim ("Deep & Narrow") ModernBERT embedding model (6 attention heads, 64 `head_dim`, 768 GeGLU intermediate size, 50,368 vocab, up to 1024–4096 context).  
**Teacher Model:** `Alibaba-NLP/gte-modernbert-base` (22 layers, 768-dim, 12 attention heads).  
**Hardware Constraint:** Single NVIDIA A100 GPU (limited VRAM and compute budget).

---

## 1. The Big Picture: Current vs. Recommended Pipeline

| Stage | Current Process | Recommended Process | Why Change? |
| :--- | :--- | :--- | :--- |
| **1. Init** | SVD `tok_embeddings` + slice **all 12 layers** | **GUIDE:** SVD `tok_embeddings` + **Layer 0 only** (Layers 1–11 random) | Initializing >1 layer traps deeper layers in rigid, sliced teacher subspaces and hurts final quality. |
| **2. Warmup / Pre-train** | Two-stage MLM (1,024 → 4,096 tokens) with `tok_embeddings` **frozen** | **Short MLM warmup** (`512` tokens) + **MiniLMv2 Relation Distillation**, all student weights **unfrozen** | Pure MLM teaches token reconstruction, not retrieval. Unfreezing lets the 384-d embeddings adapt; MiniLMv2 transfers attention structure fast. |
| **3. Dense Distill** | `MSELoss` on PCA-reduced teacher embeddings | **Cosine / InfoNCE (in-batch negatives) + MSE** using pre-computed teacher embeddings | Pure MSE only supervises Euclidean distance to a compressed point; Cosine/InfoNCE teaches ranking geometry. |
| **4. Fine-tune** | Planned (TBD) | Hard negatives / triplets (and optional long-context pass) | Specializes the model for your target retrieval domain. |

---

## 2. Deep Dive: Core Engineering Principles

### A. Why You Should NOT Freeze `tok_embeddings` (or Layer 0)
A common intuition is: *"I used SVD/GUIDE to carefully compress the teacher's embeddings—if I unfreeze them, won't training destroy that good initialization?"*

**No—freezing them is actually what hurts you.** Here is why:
1. **Teacher = Frozen, Student = Trainable:** In knowledge distillation, the *teacher* is the fixed target, so it stays frozen. The *student* needs all of its parameters unfrozen so it has full capacity to match that target.
2. **GUIDE Is a Warm Start, Not a Lock:** Think of GUIDE like starting a race 100 meters ahead of the starting line. Gradient descent doesn't erase your head start—it smoothly adapts those weights from a strong starting point toward your target objective.
3. **Compression Is Lossy:** Compressing `768 → 384` dimensions via SVD discards 50% of the teacher's capacity. If `tok_embeddings` is frozen, Layers 1–11 are forced to spend their limited capacity compensating for a rigid, lossy input space.
4. **MLM vs. Retrieval Geometry:** Embeddings initialized from a general/MLM teacher are optimized for *token prediction*, not *sentence-level semantic similarity*. Unfreezing `tok_embeddings` during distillation lets synonyms, query terms, and domain vocabulary shift closer together in 384-d space.

---

### B. Why GUIDE Initializes Only Embeddings + Layer 0
- **What Layer 0 Does:** In a Transformer, Layer 0 is the "cleanup and syntax" layer—it takes raw token embeddings and builds local syntactic/phrase representations. Initializing Layer 0 via the $M^T$ bridge gives the student immediate language-parsing ability and prevents early gradient instability / loss spikes.
- **Why Stop at Layer 0:** In the GUIDE paper (Tables 7 & 8), initializing 2, 4, 8, or `first-N` layers consistently **degraded** performance compared to initializing **Embeddings + Layer 0 only**. Deeper layers (Layers 1–11) need freedom to organize their own representations around the narrower 384-dim width rather than being forced into sliced teacher subspaces.

---

### C. Why SVD Over PCA for Embedding Compression
- **No Mean-Centering Distortion:** PCA subtracts the mean vector before decomposing, which shifts the origin of the embedding space. Transformer attention (dot products) and normalization (`LayerNorm` / `RMSNorm`) depend on the original origin and vector norms. SVD operates directly on $E_T$ without shifting the origin, preserving pairwise dot products ($E_S E_S^T \approx E_T E_T^T$).
- **Numerical Stability & Weighting:** SVD avoids materializing a $768 \times 768$ covariance matrix and naturally supports **frequency-weighted SVD** (giving common tokens higher priority than rare tokens).

---

### D. MiniLMv2: The Method vs. The Model
- **The Model:** `all-MiniLM-L6-v2` is a famous sentence-embedding model.
- **The Method (what we use here):** *MiniLMv2* (Wang et al., 2021) is a **task-agnostic distillation technique** that transfers *how the teacher attends* across tokens:
  1. **Decoupling Head Counts via Relation Heads:** Take `Q`, `K`, and `V` vectors from a single teacher layer and the student's **last layer** (layer 11). Concatenate them across attention heads and re-split into $A_r$ **relation heads** (e.g., `48` for base teachers). Because both models are reshaped to $A_r$ relation heads, your **6-head student** can learn directly from a **12-head teacher**.
  2. **The Sweet Spot (`Q-Q + K-K + V-V`):** Out of 9 possible pairings, minimizing KL-divergence on just `Q-Q`, `K-K`, and `V-V` scaled dot-product relations gives the best balance of accuracy and speed.
  3. **Upper-Middle Layer Selection:** For deep teachers (like 22-layer `gte-modernbert-base`), distill from an **upper-middle layer** (~75–85% depth, e.g., layer 17–19) rather than the final layer, which is often overly specialized for the teacher's pretraining objective.
- **Synergy with GUIDE:** The GUIDE paper proved that **GUIDE + Distillation** is nearly additive, beating either method alone.

---

### E. Why Raw `MSELoss` Alone Is Limiting in Dense Distillation
- **The Problem with Pure MSE:** Regressing only the final 384-d pooled vector with `MSELoss` forces 12 layers to guess how to route information to match a single Euclidean point. Worse, PCA-projecting the teacher's 768-d output embeddings to 384-d can distort fine-grained ranking geometry.
- **The Fix:** Pair or replace raw MSE with **Cosine Similarity / InfoNCE (with in-batch negatives)**. Semantic search cares about *relative ranking* (is Document A closer to Query Q than Document B?), not exact Euclidean coordinates.

---

## 3. Single-A100 VRAM & Optimizer Playbook

1. **Sequence Length Curriculum:** Attention memory scales quadratically ($O(L^2)$). Run your short MLM warmup + MiniLMv2 distillation at **`max_length=512`** on a single A100. Do not run teacher + student together at `4,096` tokens.
2. **Mixed Precision + Flash/SDPA Attention:** Always use `bf16=True` and `attn_implementation="sdpa"` (or `"flash_attention_2"` if installed) to avoid materializing the $L \times L$ attention matrix.
3. **Gradient Checkpointing:** Enable `gradient_checkpointing=True` (`use_reentrant=False`)—cuts activation memory by ~60–70% for ~20% extra compute.
4. **Optimizer Choice:**
   - `adamw_torch` / `adamw_torch_fused`: Both are FP32 AdamW (same VRAM; `fused` is just faster).
   - `adamw_bnb_8bit`: Quantizes optimizer states to 8-bit, **saving ~75% optimizer VRAM**—ideal for a single A100.
5. **Pre-Compute Teacher Embeddings (Stage 3):** Run `gte-modernbert-base` offline to save target embeddings to disk so the teacher doesn't occupy VRAM during dense distillation.

---

## 4. Recommended Step-by-Step Pipeline

1. **Baseline Diagnostic (Zero-Waste):** Run Phase 2 (`SentenceTransformerTrainer` with `tok_embeddings` unfrozen) on your existing Phase 1 checkpoint to get a NanoBEIR baseline.
2. **Stage 1 — GUIDE Init:** Frequency-weighted SVD on `tok_embeddings` ($768 \to 384$) + Layer 0 projection; leave Layers 1–11 randomly initialized.
3. **Stage 2 — Short MLM Warmup (`512` tokens, `bf16`, unfrozen):** Brief run to wake up Layers 1–11.
4. **Stage 3 — MiniLMv2 Relation Distillation (`512` tokens, `bf16`, unfrozen):** Distill `Q-Q + K-K + V-V` ($A_r = 48$) from `gte-modernbert-base` (layer ~17–19) into student layer 11.
5. **Stage 4 — Dense Retrieval Distillation:** Train on pre-computed `gte-modernbert-base` embeddings using **Cosine / InfoNCE + MSE**, evaluated on NanoBEIR.
6. **Stage 5 — Downstream Fine-Tuning:** Fine-tune on hard negatives / triplets (and optionally a brief long-context pass without the teacher).
