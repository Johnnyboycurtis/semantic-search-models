# Optimization & Training Specification: ModernBERT-Small (12L × 384d, GUIDE-Initialized)

This document replaces the earlier general notes on partially initialized embedding models and tailors the optimization schedule specifically to your **~37.0M-parameter `12L × 384d` ModernBERT-small** architecture, your **~20 GB GPU** budget, and your **all-12-layer GUIDE initialization** pipeline.

---

## 1. Architecture & Initialization Baseline

* **Backbone (`student_config`):** `hidden_size=384`, `num_hidden_layers=12`, `num_attention_heads=6` (64-dim heads), `intermediate_size=768` (GeGLU 4.0× pre-gate expansion), `vocab_size=50368` (reusing [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base)'s tokenizer).
* **Parameter Split:**
  * `tok_embeddings` (`50,368 × 384`): **19.34M** parameters (52.2%).
  * 12 Transformer layers (`Wqkv`, `Wo`, GeGLU `Wi`/`Wo`, norms): **17.70M** parameters (47.8%).
  * Temporary Phase 1 MLM `decoder` head (`384 × 50,368`, untied): **+19.34M** during Phase 1 only (discarded before Phase 2).
* **GUIDE Initialization (`init_model_mlm.py`):**
  * Compute the orthogonal PCA basis $M \in \mathbb{R}^{768 \times 384}$ from `ModernBERT-base`'s `tok_embeddings` using `PCA(n_components=384, svd_solver="full")` ([Guided Initialization and Distillation of Embeddings](https://arxiv.org/pdf/2510.06502)).
  * Project embeddings in the **uncentered** linear subspace ($E_S = E_T M$) so token vectors remain aligned with downstream weight projections ($W M$ and $M^T W$), and explicitly re-zero `pad_token_id` (`50283`).
  * Project **all 12 student layers** from teacher layers `[0, 1, 2, 6, 7, 8, 12, 13, 14, 19, 20, 21]`, preserving ModernBERT's alternating local sliding-window (`128` tokens) and global attention layers (`0, 3, 6, 9`).

---

## 2. Freezing Dynamics: What Changed vs. Your Earlier Notes

Your earlier notes assumed **only Layer 0** was projected while **Layers 1–11 were randomly initialized**, which required a complex two-stage MLM unfreezing schedule (Phase A1 hard freeze → Phase A2 Layer-wise Learning Rate Decay with a secondary warmup) to prevent 11 layers of random noise from destroying Layer 0 and `tok_embeddings` ([Fine-Tuning can Distort Pretrained Features](https://arxiv.org/pdf/2202.10054)).

Because your updated `init_model_mlm.py` now projects **all 12 layers**, the optimization picture is much simpler:

1. **Phase 1 MLM (`ModernBertForMaskedLM`): Freeze `tok_embeddings` only.**
   * The only uninitialized component at step 0 is the temporary MLM `decoder` head, which starts at $\approx \ln(50368) \approx 10.83$ cross-entropy loss.
   * Set `tie_word_embeddings=False` in `from_pretrained(...)` (and remove `model.tie_weights()`), warm-copy `tok_embeddings.weight` into `model.decoder.weight` once at step 0, and set `model.model.embeddings.tok_embeddings.weight.requires_grad = False`.
   * **Why:** Protecting `tok_embeddings` prevents early MLM `decoder` loss spikes from corrupting the 384-dim PCA geometry *and* eliminates gradient/optimizer-state memory for 19.34M parameters on a 20 GB GPU. Leave `embeddings.norm` and `layers[0..11]` trainable—no custom LLRD scheduler is needed.
   * **Required Trainer Flag:** Always pass `gradient_checkpointing_kwargs={"use_reentrant": False}` alongside `gradient_checkpointing=True` so frozen inputs to `layers[0]` do not drop gradients.
2. **Phase 2 & 3 Bi-Encoder (`SentenceTransformer`): Unfreeze `tok_embeddings` at a gentle LR.**
   * PCA (`768 → 384`) discards lower-variance components. Once the MLM head is discarded and the 12 transformer layers are aligned, unfreezing `tok_embeddings` during contrastive fine-tuning (`1e-5`–`2e-5`) lets the model adapt rare/domain tokens and sharpen cosine similarity margins.

---

## 3. Data Preparation Rules (Phase 1 MLM)

* **Deduplicate Flattened Triplets:** Deduplicate `MS MARCO` and `Stanford Philosophy` strings after flattening `query + positive + negative` (and drop <30-character query fragments from MLM; keep raw triplets separate for Phase 2/3).
* **Balance Corpus Mix:** Cap [HuggingFaceFW/finewiki](https://huggingface.co/datasets/HuggingFaceFW/finewiki) (e.g., 500k articles) so it does not drown out `NPR` and `Philosophy`. Consider mixing in [allenai/peS2o](https://huggingface.co/datasets/allenai/peS2o) (scientific/STEM prose) and [sentence-transformers/gooaq](https://huggingface.co/datasets/sentence-transformers/gooaq) (conversational Q&A).
* **Pack & Chunk Sequences:** Pack short passages together and chunk long articles into `1,024`-token blocks (or use `DataCollatorWithFlattening` with `attn_implementation="flash_attention_2"` per the [ModernBERT docs](https://huggingface.co/docs/transformers/model_doc/modernbert)) instead of truncating long articles or padding short rows.

---

## 4. End-to-End Optimization Schedule (20 GB GPU)

| Parameter / Setting | Phase 1A: MLM Pre-Training | Phase 1B: RoPE Extension *(Optional)* | Phase 2: Contrastive Alignment | Phase 3: Hard Negatives + MRL |
| :--- | :--- | :--- | :--- | :--- |
| **Objective** | MLM (30% masking) | MLM (30% masking) | [CachedMultipleNegativesRankingLoss](https://sbert.net/docs/sentence_transformer/loss_overview.html) | [MatryoshkaLoss](https://sbert.net/examples/sentence_transformer/training/matryoshka/README.html) (`[384, 256, 128]`) + optional `MarginMSELoss` |
| **`tok_embeddings`** | **Frozen** (`requires_grad=False`) | **Frozen** (`requires_grad=False`) | **Frozen** or `1e-5` | **Trainable** (`1e-5`–`2e-5`) |
| **Layers 0–11** | Trainable (`tie_word_embeddings=False`) | Trainable | Trainable | Trainable |
| **Sequence Length** | `1,024` tokens | `2,048–4,096` tokens (`global_rope_theta=160000.0`) | `512–1,024` (or `2,048`) | `512–1,024` (or `2,048`) |
| **Precision & Optim** | `bf16` (or `fp16` on T4), `adamw_bnb_8bit` | `bf16`, `adamw_bnb_8bit` | `bf16`, `adamw_bnb_8bit` | `bf16`, `adamw_bnb_8bit` |
| **Micro / Effective Batch** | `16` micro / `128–256` effective | `4–8` micro / `64–128` effective | `16` mini-batch / `256–512` GradCache | `16` mini-batch / `128–256` GradCache |
| **Peak Learning Rate** | `5.0e-4` (cosine, 5% warmup) | `1.0e-4` (cosine, ~10% of 1A steps) | `3.0e-5`–`5.0e-5` (cosine, 10% warmup) | `1.0e-5`–`2.0e-5` (cosine, 10% warmup) |
| **Data** | Packed deduplicated prose | Long documents only (`>1,500` tokens) | `(query, positive)` pairs (`NO_DUPLICATES`) | `(anchor, positive, hard_negative)` triplets |

### Decision Gate: When to Run Phase 1B
Check the **95th-percentile token length** of your target RAG chunks after Phase 1A completes:
* **P95 ≤ 1,024 tokens:** Skip Phase 1B and proceed directly to Phase 2.
* **P95 > 1,024 tokens:** Run Phase 1B on long documents (`FineWiki`, `NPR`, `peS2o`) before Phase 2.

---

## 5. Evaluation & Serving Checklist

* Track **NDCG@10**, **MRR@10**, and **Recall@10** after each Phase 2/3 epoch using `InformationRetrievalEvaluator` on a held-out domain corpus with hard negatives.
* Verify `Mean Pooling` + `L2 Normalization` are active in the saved `SentenceTransformer` config.
* Test truncation at `256` and `128` Matryoshka dimensions against full `384` dimensions before exporting to `float16` or ONNX INT8 for serving.
