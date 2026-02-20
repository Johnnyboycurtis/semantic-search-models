# ModernBERT-small-v2: Deep & Narrow Embedding Model

ModernBERT-small-v2 is a high-efficiency, 384-dimensional embedding model built using a rigorous four-stage pipeline. By combining structural weight transfer (GUIDE) with dense knowledge distillation, this model achieves the performance of much larger transformers while maintaining a minimal memory footprint and high storage efficiency.

## 📊 Model Specifications

| Feature | Specification |
| :--- | :--- |
| **Base Architecture** | ModernBERT (RoPE, GeGLU, Unpadding, Flash Attention) |
| **Layers** | 12 (Deep) |
| **Hidden Dimension** | 384 (Narrow/Efficient) |
| **Max Context** | 1024 Tokens |
| **Embedding Dim** | 384 |
| **Training Paradigm** | GUIDE Initialization $\to$ MLM $\to$ MSE Distillation |

---

## 🚀 The Training Pipeline

The creation of `ModernBERT-small-v2` follows a "Distill-then-Specialize" philosophy across four distinct phases:

### 1. Data Curation (Hybrid Corpus)
To ensure the model understands both casual search queries and complex academic prose, we curated a multi-domain dataset:
*   **Search & Retrieval:** MS MARCO Triplets.
*   **Academic/Philosophical:** Stanford Encyclopedia of Philosophy (SEP).
*   **General News:** NPR Paragraphs.
*   **General Web:** FineWiki (English).

### 2. Structural Initialization (The GUIDE Strategy)
Instead of starting from random weights, we used **Guided Initialization (GUIDE)** to transfer the "DNA" of `ModernBERT-base` (768-dim) into our student (384-dim).
*   **Embedding Projection:** We used PCA to project the teacher's embedding space down to 384 dimensions.
*   **Layer 0 Transfer:** The teacher’s first layer weights were projected and sliced to initialize the student, providing a "head start" on language modeling.

### 3. General Language Understanding (MLM)
The initialized model underwent **Masked Language Modeling (MLM)** on our hybrid corpus. 
*   **Objective:** Re-align the transferred weights to the new 12-layer depth and 384-width.
*   **Masking:** Increased masking probability (30%) to force deeper semantic reasoning.
*   **Outcome:** A robust base model that understands context, grammar, and domain-specific terminology.

### 4. Dense Knowledge Distillation (The GTE Strategy)
The final stage transforms the MLM model into a retrieval expert by distilling knowledge from `Alibaba-NLP/gte-modernbert-base`.
*   **Teacher:** GTE-ModernBERT (768-dim).
*   **Projection:** A PCA-based linear layer was added to the teacher to map its 768-dim outputs exactly to the student's 384-dim space.
*   **Loss Function:** **MSE (Mean Squared Error)**. The student is trained to minimize the distance between its vector output and the teacher's projected output.
*   **Evaluation:** Monitored via `NanoBEIR` (MSMARCO & HotpotQA) to ensure retrieval accuracy.

---

## 🔬 Research & Philosophical Basis

This project synthesizes methodologies from several landmark papers in the field of efficient transformer training:

| Concept | Source Research | Implementation in v2 |
| :--- | :--- | :--- |
| **Deep & Narrow** | *Deep & Narrow Transformers* | Moving from 6 wide layers to 12 narrow layers for better semantic reasoning. |
| **GUIDE** | *Google Research* | Using PCA and Weight Projection to initialize the student from a 768-dim teacher. |
| **Score Distillation** | *IBM Granite / LEAF* | Regressing the student's embeddings directly to a high-performance teacher's output space. |
| **Modern Architecture** | *HuggingFace ModernBERT* | Native support for RoPE (Rotary Positional Embeddings) and GeGLU for superior 1024+ context handling. |
| **Data Filtering** | *Nomic AI / Gecko* | Leveraging GTE-ModernBERT to provide "soft labels" (Teacher targets) for distillation. |

---

## 🛠️ Usage

This model is designed for use with `sentence-transformers`:

```python
from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer("johnnyboycurtis/ModernBERT-small-v2")

# Encode text
sentences = ["The philosophical implications of AI.", "How to train a ModernBERT model."]
embeddings = model.encode(sentences)

print(embeddings.shape) # (2, 384)
```

## ⚖️ License
This project is released under the Apache 2.0 License. Use of the training data is subject to the original licenses of MS MARCO, SEP, NPR, and FineWiki.



---

This set of scripts documents a sophisticated, multi-stage workflow designed to create a specialized, highly efficient Transformer model, `ModernBERT-small-v2`, targeting 384-dimensional embeddings.

The process involves **Data Curation**, **GUIDE Initialization** (transferring structural knowledge), **MLM Pre-training** (general language understanding), and final **Knowledge Distillation** (mapping to the target embedding space).

---

## 1. Workflow Overview

The creation process is divided into four distinct phases executed sequentially:

| Phase | Script | Goal | Key Technique |
| :--- | :--- | :--- | :--- |
| **Phase 1** | `mlm/create_dataset.py` | Data Curation | Combining four diverse sources (Search, Philosophy, News, Wiki) into one MLM corpus. |
| **Phase 2** | `mlm/init_model_mlm.py` | Model Initialization | Transferring weights from a large teacher model into the deep/narrow student configuration using **GUIDE**. |
| **Phase 3** | `mlm/pre_train.py` | MLM Pre-Training | Completing the pre-training of the student model structure using the curated corpus. |
| **Phase 4** | `create_distil_dataset.py` | Data Curation | Combining four diverse sources (Search, Philosophy, News, Wiki, FineWiki) into one corpus. |
| **Phase 5** | `distil_train.py` | Knowledge Distillation | Training the MLM-ready student model against a powerful retrieval teacher (`gte-modernbert-base`) to optimize for dense vector similarity (STS). |

---

The creation of **ModernBERT-small-v2** follows a sophisticated "Cold-Start to Expert" pipeline. Instead of training a small model from scratch—which is computationally expensive and often results in lower performance—this process uses structural weight transfer and multi-stage distillation.

Here is a detailed breakdown of the five phases of the training process.

---

### Phase 1: Data Curation (`mlm/create_dataset.py`)
**Goal:** Build a "knowledge-rich" corpus that covers diverse linguistic styles.

Before training begins, we aggregate four high-quality data sources to ensure the model isn't biased toward a single domain:
*   **MS MARCO:** Provides "search-style" queries and web snippets.
*   **Stanford Encyclopedia of Philosophy (SEP):** Introduces complex, high-level academic reasoning and logical structures.
*   **NPR:** Offers clean, journalistic prose with a focus on narrative structure.
*   **FineWiki (English):** Provides a massive baseline of general web knowledge.

**The Technique:** All datasets are "flattened." Triplets (Query/Positive/Negative) are broken down into individual text strings, and long documents are chunked. The result is a unified Parquet file containing millions of rows of clean, diverse text.

---

### Phase 2: Model Initialization (`mlm/init_model_mlm.py`)
**Goal:** To inherit the structural intelligence of a larger model.

We define a **Deep & Narrow** architecture: 12 layers deep (to allow for complex multi-hop reasoning) but only 384 dimensions wide (to keep the model fast and the storage footprint small).

**The Technique (GUIDE):**
Instead of random initialization, we use **Guided Initialization (GUIDE)** from the `ModernBERT-base` teacher:
1.  **Embedding PCA:** We take the teacher’s 768-dim embeddings and use Principal Component Analysis (PCA) to project them down to 384-dim. This preserves the semantic relationships the teacher already learned.
2.  **Layer Projection:** We take the teacher’s first layer and project its weights into the student’s dimensions.
3.  **Structural Transfer:** By initializing the student with these projected weights, the model starts Phase 3 already understanding basic syntax and word relationships.

---

### Phase 3: MLM Pre-Training (`mlm/pre_train.py`)
**Goal:** Adapt the initialized "skeleton" to the specific 12-layer depth and 384-width.

While Phase 2 gave us a head start, the model's internal layers (1–11) are still effectively blank. We use **Masked Language Modeling (MLM)** to train the model to predict hidden words.

**The Technique:**
*   **High Masking Rate:** We use a **30% masking probability** (higher than the standard 15%). This makes the "reconstruction" task harder, forcing the model to learn deeper context rather than just local word patterns.
*   **ModernBERT Features:** We leverage RoPE (Rotary Positional Embeddings) and GeGLU activations, allowing the model to handle contexts up to **1024 tokens** efficiently.
*   **Result:** This produces a "Base" model that is a general-purpose language expert but not yet a retrieval expert.

---

### Phase 4: Distillation Dataset Prep (`create_distil_dataset.py`)
**Goal:** Generate "Expert Labels" for the student to follow.

To turn a Language Model into an Embedding Model, we need a teacher that is already an expert at retrieval. We use `Alibaba-NLP/gte-modernbert-base`.

**The Technique:**
1.  **Dimensionality Alignment:** The GTE teacher outputs 768-dim vectors, but our student only has 384. We encode a sample of the data, fit a PCA, and add a **Dense Projection Layer** to the teacher.
2.  **Soft-Labeling:** We run the entire curated corpus from Phase 1 through this "Reduced Teacher."
3.  **Outcome:** We create a new dataset where every sentence is paired with a high-quality 384-dim vector. The student no longer has to guess what a "good" embedding looks like; it has a direct map to follow.

---

### Phase 5: Knowledge Distillation (`distil_train.py`)
**Goal:** Final specialization for Semantic Textual Similarity (STS) and Retrieval.

This is the final "finishing school" for the model. We move away from predicting words (MLM) and move toward predicting **vector space coordinates**.

**The Technique:**
*   **MSE Loss (Mean Squared Error):** The student model encodes a sentence, and the training objective is to minimize the distance between the student’s vector and the teacher’s "expert" vector from Phase 4.
*   **Continuous Evaluation:** During training, we use **NanoBEIR**. This checks the model’s performance on real-world retrieval tasks (MS MARCO and HotpotQA) every 1,000 steps.
*   **Convergence:** Because the student was already well-initialized (GUIDE) and pre-trained (MLM), it converges to the teacher's performance very quickly.

**The Final Result:** `ModernBERT-small-v2`—a model that has the "brain" of a 12-layer transformer, the "efficiency" of a 384-dim vector, and the "retrieval accuracy" distilled from the SOTA GTE family.