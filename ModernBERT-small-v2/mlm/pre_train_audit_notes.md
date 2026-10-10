# Technical Audit & Pre-Training Notes: ModernBERT-small-v2

This document records the architectural, methodological, and optimization issues identified during the review of [`pre_train.py`](file:///home/jonathan/Projects/semantic-search-models/ModernBERT-small-v2/mlm/pre_train.py) and [`pre_train.md`](file:///home/jonathan/Projects/semantic-search-models/ModernBERT-small-v2/mlm/pre_train.md), along with actionable remediation strategies.

---

## Issue 1: Context Window & Tokenization Mismatch
*Treating MLM text as isolated short snippets instead of continuous, packed 1024-token streams.*

### Problem Description
In [`pre_train.md`](file:///home/jonathan/Projects/semantic-search-models/ModernBERT-small-v2/mlm/pre_train.md#L178-L181) (Section 5), it is argued that:
> *"The script treats every line in your data as a standalone paragraph or sentence. Why: This model’s ultimate destiny is to become a Sentence-Transformer... By training it on discrete lines of text now, we are preparing its 'brain' for the specific way it will be used later in semantic search."*

In [`pre_train.py`](file:///home/jonathan/Projects/semantic-search-models/ModernBERT-small-v2/mlm/pre_train.py#L51-L66), sentences and queries from `combined_mlm_dataset.parquet` are tokenized individually with `truncation=True, max_length=1024`.

### Failure Mechanics & Theoretical Impact
1. **Context Window Underutilization:** MS MARCO queries average ~6 tokens, and passages average ~50 tokens. Feeding them as discrete lines means 90–99% of the allocated 1,024-token context capacity is wasted on padding tokens.
2. **Alternating Sliding Window Attention Bypassed:** ModernBERT uses 128-token local sliding window attention on 2 out of every 3 layers. When sequences are shorter than 128 tokens, the sliding window spans the entire text, effectively collapsing the local attention mechanism into trivial global attention.
3. **Failure to Learn Long-Range Bidirectional Reasoning:** MLM pre-training is fundamentally designed to learn discourse coherence, cross-sentence syntax, and long-range semantic dependencies. Sentence-level retrieval formatting belongs in the downstream distillation/fine-tuning phase, not the base MLM pre-training phase.

### Remediation
* **Grouped Text Concatenation:** Concatenate tokens across examples with `<eos>` or `[SEP]` delimiters and chunk into uniform blocks of 1,024 tokens (standard in RoBERTa and Hugging Face's `run_mlm.py`).
* **Sequence Packing:** Implement unpadded greedy sequence packing as described in the ModernBERT paper (Warner et al., Section 2.2.1) to achieve >99% packing efficiency without padding waste.

---

## Issue 2: Gradient Shock from Cold-Started Upper Layers (Layers 1–11)
*Violent backpropagation from random weights destabilizing pre-initialized layers.*

### Problem Description
The model architecture uses a hybrid state: Embeddings and Layer 0 are initialized via GUIDE, while Layers 1–11 and the Masked LM head are completely random Gaussian weights. Training is started with a high learning rate ($5\times 10^{-4}$ in code, $8\times 10^{-4}$ in documentation) and a short 1,000-step warmup.

### Failure Mechanics & Theoretical Impact
1. **High Initial Loss:** At Step 0, predictions from random upper layers produce uniform entropy over the 50,368-token vocabulary ($\text{loss} \approx -\ln(1/50368) \approx 10.8$).
2. **Chaotic Gradient Backpropagation:** The resulting massive gradient signals flow backward through 11 layers of random matrices into Layer 0 and Embeddings.
3. **Weight Disruption:** Because the learning rate is high and the warmup is brief, these uncalibrated early gradients violently distort the weights of Layer 0 and Embeddings before Layers 1–11 have learned basic feature representations.

### Remediation
* **Staged Warmup:** Warm up the random layers while keeping early layers constrained.
* **Discriminative Layer-Wise Learning Rates (LLRD):** Apply a significantly lower learning rate (e.g., $0.1\times$ to $0.05\times$) to Layer 0 and Embeddings compared to Layers 1–11.
* **Freezing (see Issue 4):** Completely freeze Layer 0 and Embeddings during the initial pre-training stages.

---

## Issue 3: LM Head Missing Initialization & Tied Weights Handling
*Decoupling or clobbering tied word embeddings when instantiating `ModernBertForMaskedLM`.*

### Problem Description
[`init_model_mlm.py`](file:///home/jonathan/Projects/semantic-search-models/ModernBERT-small-v2/mlm/init_model_mlm.py#L73) initializes and saves a `ModernBertModel` (backbone only). When [`pre_train.py`](file:///home/jonathan/Projects/semantic-search-models/ModernBERT-small-v2/mlm/pre_train.py#L33) loads this checkpoint using `AutoModelForMaskedLM.from_pretrained`, Hugging Face warns:
> *"Some weights of ModernBertForMaskedLM were not initialized from the model checkpoint... [decoder.weight, decoder.bias, head.dense.weight, head.norm.weight]"*

In [`pre_train.md`](file:///home/jonathan/Projects/semantic-search-models/ModernBERT-small-v2/mlm/pre_train.md#L145-L148), this warning was accepted as desirable:
> *"This is expected. Your GUIDE script built the 'Backbone.' The Training script is now building the 'Head' on top of it. This is exactly what we want."*

### Failure Mechanics & Theoretical Impact
1. **Tied Embeddings Inconsistency:** ModernBERT sets `tie_word_embeddings = True`, meaning `model.model.embeddings.tok_embeddings.weight` and `model.decoder.weight` share the exact same tensor pointer in memory.
2. **Random Overwrite Risk:** When `from_pretrained` detects missing checkpoint keys for `decoder.weight`, it randomly initializes them. Depending on when `model.tie_weights()` is executed and how the loader resolves missing keys, this either leaves the output projection untied or overwrites the PCA-projected input embeddings with Gaussian noise.
3. **Discarding Teacher's LM Head:** The teacher model (`ModernBERT-base`) already possesses a pre-trained prediction head (`head.dense`, `head.norm`). Forcing the student to learn vocabulary prediction from scratch through a random dense head creates unnecessary training volatility.

### Remediation
* **Initialize `ModernBertForMaskedLM` Directly:** In [`init_model_mlm.py`](file:///home/jonathan/Projects/semantic-search-models/ModernBERT-small-v2/mlm/init_model_mlm.py), instantiate the student as `ModernBertForMaskedLM(student_config)`.
* **Project Teacher Head:** Apply PCA projection $M$ to `teacher.head.dense.weight` ($768 \to 384$) or properly tie `decoder.weight` to `tok_embeddings.weight` prior to saving.

---

## Issue 4: Unfrozen Embeddings & Layer 0 Overwritten (Loss of GUIDE Faithfulness)
*Failing to freeze transferred teacher parameters, leading to catastrophic anchor drift.*

### Problem Description
During the pre-training loop in [`pre_train.py`](file:///home/jonathan/Projects/semantic-search-models/ModernBERT-small-v2/mlm/pre_train.py), both the PCA-compressed token embedding table ($E_S = E_T M$) and the projected Layer 0 weights ($W_{S,0} \approx M^T W_{T,0}$) retain `requires_grad = True` by default.

### Failure Mechanics & Theoretical Impact
1. **Loss of the GUIDE Semantic Bridge:** The core thesis of the GUIDE paper (Trinh, Menghani & Vee, Google Research, 2025) is that $M$ serves as an exact mathematical bridge connecting the student's lower-dimensional space to the teacher's manifold. Updating $E_S$ and Layer 0 from Step 0 allows them to rapidly drift away from the teacher's coordinate system.
2. **Accommodation of Random Error:** Rather than Layers 1–11 being forced to adapt to the teacher's syntactic and semantic foundation, the optimizer updates the foundation to satisfy the random noise of the upper layers. GUIDE effectively degrades into a brief, rapidly overwritten warm start.
3. **Massive Memory & Compute Penalty:** ModernBERT's vocabulary table is $50{,}368 \times 384 \approx 19.34\text{M}$ parameters. Leaving it trainable forces AdamW/StableAdamW to maintain momentum and variance buffers for all 19.34M parameters, adding hundreds of megabytes of VRAM overhead and slowing the backward pass.

### Remediation
Explicitly freeze the embeddings and Layer 0 in [`pre_train.py`](file:///home/jonathan/Projects/semantic-search-models/ModernBERT-small-v2/mlm/pre_train.py) before passing the model to `Trainer`:

```python
# -------------------------------------------------------------
# Preserve GUIDE Foundation: Freeze Embeddings and Layer 0
# -------------------------------------------------------------
# 1. Freeze input token embeddings
for param in model.model.embeddings.parameters():
    param.requires_grad = False

# 2. Freeze Layer 0 (transferred logic gate)
for param in model.model.layers[0].parameters():
    param.requires_grad = False

print("🔒 Successfully frozen Embeddings and Layer 0 to preserve GUIDE subspace.")
```

#### Staged Unfreezing Alternative (Optional)
If domain adaptation of the vocabulary is desired:
1. **Phase 3a (Steps 0 – 10,000):** Keep Embeddings and Layer 0 strictly **frozen** while Layers 1–11 converge from loss $\approx 10.8$ down to $\approx 3.0$.
2. **Phase 3b (Annealing / Final steps):** Unfreeze Embeddings and Layer 0 with a reduced learning rate ($10\times$ lower LR than upper layers) for fine-tuning.
