# Technical Audit & Pre-Training Notes: ModernBERT-small-v2

> **Document Status:** Active Audit / Remediation Proposals  
> **Target Scripts:** [`pre_train.py`](file:///home/jonathan/Projects/semantic-search-models/ModernBERT-small-v2/mlm/pre_train.py), [`pre_train.md`](file:///home/jonathan/Projects/semantic-search-models/ModernBERT-small-v2/mlm/pre_train.md), [`init_model_mlm.py`](file:///home/jonathan/Projects/semantic-search-models/ModernBERT-small-v2/mlm/init_model_mlm.py)  
> **Related Reference:** [`partially_initialized_embedding_models_notes.md`](file:///home/jonathan/Projects/semantic-search-models/ModernBERT-small-v2/mlm/partially_initialized_embedding_models_notes.md)  
> **Note:** The issues documented below are tracked audit findings. Comprehensive remediation strategies are provided for each, but the underlying codebase remains unaltered pending review and testing.

---

## Issue 1: Context Window & Tokenization Mismatch
*Treating MLM text as isolated short snippets instead of continuous, packed 1,024-token streams.*

* **Status:** `OPEN (Pending Remediation)`
* **Severity:** High (Throughput penalty & architectural invalidation)

### Problem Description
In [`pre_train.md`](file:///home/jonathan/Projects/semantic-search-models/ModernBERT-small-v2/mlm/pre_train.md#L178-L181) (Section 5), it is argued:
> *"The script treats every line in your data as a standalone paragraph or sentence. Why: This model’s ultimate destiny is to become a Sentence-Transformer... By training it on discrete lines of text now, we are preparing its 'brain' for the specific way it will be used later in semantic search."*

In [`pre_train.py`](file:///home/jonathan/Projects/semantic-search-models/ModernBERT-small-v2/mlm/pre_train.py#L51-L66), sentences and queries from `combined_mlm_dataset.parquet` are tokenized individually with `truncation=True, max_length=1024`.

### Failure Mechanics & Theoretical Impact
1. **Severe Context Window Underutilization:** MS MARCO queries average ~6 tokens, and passages average ~50 tokens. Feeding them as discrete lines means 90–99% of the allocated 1,024-token capacity is wasted on padding tokens.
2. **Alternating Sliding Window Attention Bypassed:** ModernBERT uses 128-token local sliding window attention on 2 out of every 3 layers. When sequences are shorter than 128 tokens, the sliding window spans the entire text, effectively collapsing the local attention mechanism into trivial global attention.
3. **Failure to Learn Long-Range Bidirectional Reasoning:** MLM pre-training is designed to learn discourse coherence, cross-sentence syntax, and long-range semantic dependencies. Sentence-level retrieval formatting belongs in the downstream distillation/fine-tuning phase, not the base MLM pre-training phase.

### Proposed Remediation & Implementation Suggestions
* **Suggestion 1A (Grouped Text Concatenation):**
  Concatenate tokens across examples with `<eos>` / `[SEP]` delimiters and chunk into uniform blocks of 1,024 tokens (the standard protocol in RoBERTa and Hugging Face's `run_mlm.py`):
  ```python
  def group_texts(examples, block_size=1024):
      concatenated = {k: sum(examples[k], []) for k in examples.keys()}
      total_length = len(concatenated[list(examples.keys())[0]])
      total_length = (total_length // block_size) * block_size
      return {
          k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
          for k, t in concatenated.items()
      }
  ```
* **Suggestion 1B (Unpadded Sequence Packing):**
  Implement unpadded greedy sequence packing as described in ModernBERT (Warner et al., Section 2.2.1), concatenating jagged sequences up to 1,024 tokens to achieve >99% packing efficiency without padding waste.

---

## Issue 2: Gradient Shock from Cold-Started Upper Layers (Layers 1–11)
*Violent backpropagation from random weights destabilizing pre-initialized layers.*

* **Status:** `OPEN (Pending Remediation)`
* **Severity:** Critical (Optimization stability & representation preservation)

### Problem Description
The model architecture uses a hybrid state: Embeddings and Layer 0 are initialized via GUIDE, while Layers 1–11 and the Masked LM head are completely random Gaussian weights. Training is started with a high learning rate ($5\times 10^{-4}$ in code, $8\times 10^{-4}$ in documentation) and a short 1,000-step warmup.

### Failure Mechanics & Theoretical Impact
1. **High Initial Cross-Entropy Loss:** At Step 0, predictions from random upper layers produce uniform entropy over the 50,368-token vocabulary ($\text{loss} \approx -\ln(1/50368) \approx 10.83$).
2. **Chaotic Gradient Backpropagation:** The resulting massive gradient signals flow backward through 11 layers of random matrices into Layer 0 and Embeddings.
3. **Feature Distortion:** Grounded in the LP-FT (Linear Probing then Fine-Tuning) framework (Kumar et al., ICLR 2022), early gradients do not represent structural semantic corrections; they reflect high-variance noise trying to counteract uncalibrated upper layers. In AdamW, large gradient magnitudes distort the moment vectors ($m_t$ and $v_t$), rapidly overwriting the transferred orthogonal subspace.

### Proposed Remediation & Implementation Suggestions
* **Suggestion 2A (Linear Probing then Fine-Tuning / Two-Stage LP-FT):**
  Hold the pre-trained feature base ($E_S$ and Layer 0) static while randomly initialized parameters minimize task loss from $\sim 10.8$ down to convergence or stability ($\sim 3.0\text{--}3.5$), followed by holistic fine-tuning.
* **Suggestion 2B (Layer-wise Learning Rate Decay - LLRD):**
  Scale the learning rate multiplicatively by depth ($\eta_l = \eta_{\text{base}} \cdot \gamma^{L-l}$, with $\gamma \approx 0.82$), ensuring lower layers receive an order-of-magnitude lower learning rate than top layers.

---

## Issue 3: LM Head Missing Initialization & Tied Weights Handling
*Decoupling or clobbering tied word embeddings when instantiating `ModernBertForMaskedLM`.*

* **Status:** `OPEN (Pending Remediation)`
* **Severity:** High (Structural consistency & loss dynamics)

### Problem Description
[`init_model_mlm.py`](file:///home/jonathan/Projects/semantic-search-models/ModernBERT-small-v2/mlm/init_model_mlm.py#L73) initializes and saves a `ModernBertModel` (backbone only). When [`pre_train.py`](file:///home/jonathan/Projects/semantic-search-models/ModernBERT-small-v2/mlm/pre_train.py#L33) loads this checkpoint using `AutoModelForMaskedLM.from_pretrained`, Hugging Face warns:
> *"Some weights of ModernBertForMaskedLM were not initialized from the model checkpoint... [decoder.weight, decoder.bias, head.dense.weight, head.norm.weight]"*

In [`pre_train.md`](file:///home/jonathan/Projects/semantic-search-models/ModernBERT-small-v2/mlm/pre_train.md#L145-L148), this warning was accepted as desirable:
> *"This is expected. Your GUIDE script built the 'Backbone.' The Training script is now building the 'Head' on top of it. This is exactly what we want."*

### Failure Mechanics & Theoretical Impact
1. **Tied Embeddings Inconsistency:** ModernBERT sets `tie_word_embeddings = True`, meaning `model.model.embeddings.tok_embeddings.weight` and `model.decoder.weight` share the exact same tensor pointer in memory.
2. **Random Overwrite Risk:** When `from_pretrained` detects missing checkpoint keys for `decoder.weight`, it randomly initializes them. Depending on when `model.tie_weights()` is executed and how the loader resolves missing keys, this either leaves the output projection untied or risks overwriting the PCA-projected input embeddings with Gaussian noise.
3. **Discarding Teacher's LM Head:** The teacher model (`ModernBERT-base`) already possesses a pre-trained prediction head (`head.dense`, `head.norm`). Forcing the student to learn vocabulary prediction from scratch through a random dense head creates unnecessary training volatility.

### Proposed Remediation & Implementation Suggestions
* **Suggestion 3A (Initialize as `ModernBertForMaskedLM` Directly):**
  In [`init_model_mlm.py`](file:///home/jonathan/Projects/semantic-search-models/ModernBERT-small-v2/mlm/init_model_mlm.py), instantiate the student directly as `ModernBertForMaskedLM(student_config)`.
* **Suggestion 3B (Project and Transfer Teacher's Prediction Head):**
  Apply the PCA projection matrix $M$ to `teacher.head.dense.weight` ($768 \to 384$) and copy `teacher.head.norm` weights, giving the prediction head an informed starting point.
* **Suggestion 3C (Enforce Tied Weights Before Checkpointing):**
  Explicitly verify `student.tie_weights()` and check that `student.model.embeddings.tok_embeddings.weight is student.decoder.weight` before calling `save_pretrained()`.

---

## Issue 4: Unfrozen Embeddings & Layer 0 Overwritten (Loss of GUIDE Faithfulness)
*Failing to manage parameter updates on transferred teacher weights during pre-training.*

* **Status:** `OPEN (Pending Remediation)`
* **Severity:** Critical (Core thesis of the project)

### Problem Description
During the pre-training loop in [`pre_train.py`](file:///home/jonathan/Projects/semantic-search-models/ModernBERT-small-v2/mlm/pre_train.py), both the PCA-compressed token embedding table ($E_S = E_T M$) and the projected Layer 0 weights ($W_{S,0} \approx M^T W_{T,0}$) retain `requires_grad = True` from Step 0.

### Failure Mechanics & Theoretical Verification
1. **What the GUIDE Paper (Trinh et al., 2025) Actually Did:**
   * In the original paper, GUIDE was introduced strictly as a **parameter initialization** technique (`Algorithm 2 GUIDE(S, T)`). All student parameters remained trainable.
   * *Why it worked in the paper:* The authors evaluated on **400M and 1B parameter students pre-trained on 20 Billion tokens from C4**. Over 20B tokens, initial gradient noise is amortized over hundreds of thousands of steps, allowing all parameters to co-adapt.
   * *Why it fails on smaller domain runs:* Our student is **384-dimensional (~40M parameters)** trained on $\sim 100\text{M}\text{--}200\text{M}$ tokens (100× fewer tokens). The initial gradient shock cannot be amortized; it permanently degrades the transferred weights during the earliest steps.
2. **The Danger of Permanent Hard Freezing:**
   * Truncating from 768 to 384 dimensions via PCA discards secondary variance components. If $E_S$ is frozen permanently across all stages, the model cannot rotate or adjust its coordinate axes to compensate for omitted singular vectors or domain-specific vocabulary nuances.
3. **Memory & Compute Penalty:**
   * The embedding table ($50{,}368 \times 384$) contains $\approx 19.34\text{M}$ parameters. Leaving it trainable from Step 0 forces AdamW to maintain momentum and variance buffers for all 19.34M parameters, adding hundreds of megabytes of unnecessary VRAM overhead.

### Proposed Remediation & Phased Training Schedule
As formalized in [`partially_initialized_embedding_models_notes.md`](file:///home/jonathan/Projects/semantic-search-models/ModernBERT-small-v2/mlm/partially_initialized_embedding_models_notes.md), the recommended solution is an integrated **3-phase curriculum**:

```
[GUIDE Init] ──► Phase A1: Alignment Warmup ──► Phase A2: Joint Adaptation ──► Phase B: Dense Distillation
                  (Steps 0 - 5,000)             (Steps 5,000+)                 (Sentence-Transformers)
                  • Embeddings: FROZEN          • Embeddings: TRAINABLE        • All layers: TRAINABLE
                  • Layer 0: FROZEN             • LLRD (γ = 0.82)              • Base LR = 1e-4
                  • Layers 1-11: TRAINABLE      • Secondary Warmup on E_S      • MSELoss vs GTE Teacher
                  • LR = 5e-4                   • Base LR = 3e-4
```

#### Detailed Phase Specifications

| Training Phase | Step Range | $E_S$ & Layer 0 State | Layers 1–11 State | Peak Learning Rate | Schedule Details |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase A1: Alignment Warmup** | Steps 0 – 5,000 | **Frozen** (`requires_grad = False`) | Trainable (`requires_grad = True`) | $\eta_{\text{top}} = 5 \times 10^{-4}$ | Linear warmup (2,000 steps) to constant; absorbs initial $10.8$ loss spike until loss reaches $\approx 3.0\text{--}3.5$. |
| **Phase A2: Joint Subspace Adaptation** | Steps 5,000 – End of MLM | **Trainable** (`requires_grad = True`) | Trainable (`requires_grad = True`) | $\eta_{\text{base}} = 3 \times 10^{-4}$; $\eta_{E_S} \approx 2.5 \times 10^{-5}$ | LLRD with $\gamma = 0.82$; secondary 2,000-step linear warmup on $E_S$ (Steps 5k–7k) to prevent boundary shock, followed by cosine decay. |
| **Phase B: Dense Distillation** | Full Distillation Run | **Trainable** (`requires_grad = True`) | Trainable (`requires_grad = True`) | $\eta = 1 \times 10^{-4}$ | Standard linear warmup (1,000 steps) with cosine decay to $1 \times 10^{-6}$; full end-to-end MSE distillation against `gte-modernbert-base`. |

---

## Action Plan & Execution Roadmap (Pending Implementation)

1. [ ] **Refactor Initialization Script ([`init_model_mlm.py`](file:///home/jonathan/Projects/semantic-search-models/ModernBERT-small-v2/mlm/init_model_mlm.py)):**
   - Instantiate directly as `ModernBertForMaskedLM`.
   - Project and initialize the teacher prediction head.
   - Enforce explicit tied weight tensor identity before saving.
2. [ ] **Restructure Pre-training Dataset Pipeline ([`create_dataset.py`](file:///home/jonathan/Projects/semantic-search-models/ModernBERT-small-v2/mlm/create_dataset.py)):**
   - Implement grouped token stream concatenation and chunking to 1,024-token blocks.
3. [ ] **Split Pre-training Execution into Two Staged Scripts:**
   - **`mlm/pre_train_warmup.py` (Phase A1):** Freezes $E_S$ and Layer 0, runs 5,000 steps to stabilize upper layers.
   - **`mlm/pre_train_joint.py` (Phase A2):** Loads checkpoint from Phase A1, unfreezes all layers with LLRD parameter groups, and trains to completion.
4. [ ] **Proceed to Phase B Distillation ([`distil_train.py`](file:///home/jonathan/Projects/semantic-search-models/ModernBERT-small-v2/distil_train.py)):**
   - Train the full student backbone end-to-end against `Alibaba-NLP/gte-modernbert-base`.
