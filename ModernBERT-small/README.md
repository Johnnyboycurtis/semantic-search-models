# ModernBERT-small: A Principled Approach to Efficient Sentence Embeddings

This repository details the development and training of `ModernBERT-small`, a custom, small-scale variant of the ModernBERT architecture specifically designed for generating high-quality sentence embeddings. Our approach emphasizes principled downscaling, multi-task learning, and the use of modern loss functions to achieve a cost-effective yet high-performance retrieval model.

## Project Structure & Training Workflow

The project is organized into distinct steps, allowing for modular development and clear understanding of the training pipeline:

1.  **Build and Save Blank Model:**
    *   Run `python build_and_save_small_modernbert.py`
    *   This script initializes the `ModernBERT-small` architecture with random weights and saves it, providing a clean starting point for pretraining.

2.  **Pretrain the Model:**
    *   Run `python pre_train_small_modernbert.py`
    *   This is the core pretraining script, leveraging a multi-task, multi-loss setup on a diverse collection of datasets to build a general-purpose sentence embedding model.

3.  **Fine-tune the Model:**
    *   Run `python fine_tune_small_modernbert.py`
    *   This script further fine-tunes the pretrained model specifically for information retrieval tasks, utilizing a multi-dataset strategy and real-world BEIR benchmarks for evaluation.

4.  **Benchmark (Optional):**
    *   Run `python benchmark_hotpotqa.py`
    *   (Sample; full dataset takes ~2 hours) This script can be used to evaluate the model's performance on a specific retrieval task like HotpotQA.

---

## ModernBERT-small: Architectural Design Rationale

The `ModernBERT-small` configuration represents a successful and deliberate engineering effort to create a specialized, efficient variant of the ModernBERT architecture. The goal was not simply to create a smaller model, but to perform a **principled downscaling** of the official `ModernBERT-base` architecture. By adhering to the core design philosophy and scaling rules presented in the original paper ["Smarter, Better, Faster, Longer"](https://arxiv.org/abs/2412.13663), we ensure architectural consistency, creating a smaller, highly efficient model that is a true member of the ModernBERT family.

The resulting model provides a cost-effective and capable foundation for building a high-performance retrieval model, striking a balance between performance and computational cost:

```
      ▲
      |
 High |                   /----- (Large Models: High cost, high performance)
      |                  /
 Perf.|                 /
      |        /------- (Medium Models: Balanced)
      |       /
      |      /
      | YOU ARE HERE
      |   /-- (Small Models: Low cost, good performance)
      |  /
 Low  | /
      +----------------------------------------------------►
        Low                    Cost / Latency                 High
```

The final configuration is as follows:

```python
from transformers import ModernBertConfig

modernbert_small_config = ModernBertConfig(
    hidden_size=384,
    num_hidden_layers=6,
    num_attention_heads=6,
    intermediate_size=576,
    max_position_embeddings=1024,
    hidden_activation="gelu",
)
```

Below is a parameter-by-parameter verification against the design notes in the paper:

#### Key Architectural Parameters

*   **`hidden_size: 384`**
    This represents a standard 2x reduction from the `base` model's hidden size of 768. It's a conventional choice for a "small" variant that significantly reduces memory and computational requirements while maintaining enough capacity for strong performance.

*   **`num_hidden_layers: 6`**
    The `base` model features 22 layers, following a "Deep & Narrow" design paradigm. For a "small" variant, a substantial reduction is required to lower the parameter count and accelerate inference. 6 layers provides a good balance, offering sufficient depth for representation learning while fitting the efficiency goals of a small model.

*   **`num_attention_heads: 6`**
    This value was derived to maintain a consistent attention head dimension, a crucial factor for model stability and performance.
    *   **Base Model:** `hidden_size=768` / `num_attention_heads=12` = **64 dimensions per head**.
    *   **Small Model:** `hidden_size=384` / `num_attention_heads=6` = **64 dimensions per head**.
    By keeping the head dimension constant, we preserve a key architectural property of the original model.

#### The `intermediate_size` and GeGLU Expansion Ratio (A Deeper Look)

The calculation for `intermediate_size` is the most critical part of this downscaling, as it directly relates to the model's FFN (Feed-Forward Network) layers and its use of the GeGLU activation.

In any Transformer model, the two most important components are the Self-Attention mechanism and the Feed-Forward Network (FFN). While attention handles the mixing of information across the sequence, the FFN is responsible for the deep processing of information at each token position independently. The size and structure of this FFN block are critical to the model's overall capacity and performance.

ModernBERT specifically uses **GeGLU**, which is a Gated Linear Unit (GLU) variant where the activation function is GELU. A GLU introduces a **data-dependent gating mechanism** into the FFN, allowing the model to dynamically control which information is most important to preserve. This is achieved by splitting the single "up-projection" into two parallel up-projections, one of which acts as a "gate."

**Connecting GLU Structure to the `intermediate_size` Parameter:**

1.  **Verifying the Expansion Ratio from the Paper:**
    *   According to **Table 4** in the paper, `ModernBERT-base` has a `hidden_size` of 768 and a `GLU Expansion` of 2,304.
    *   This reveals the core FFN expansion ratio: `2,304 / 768` = **3.0x**.
    *   The paper also shows that the `intermediate_size` for the base model is `1,152`. This confirms the implementation detail for GeGLU: the `intermediate_size` parameter defines the size of one of the two gating layers, meaning the total expansion is `intermediate_size * 2`. Indeed, `1,152 * 2 = 2,304`.

2.  **Applying this Ratio to `ModernBERT-small`:**
    *   With a `hidden_size` of `384`, we first calculate the target total FFN expansion: `384 * 3.0 = 1,152`.
    *   To implement this with a GeGLU architecture, we must set the `intermediate_size` to be half of the total expansion.
    *   Therefore, the correct `intermediate_size` is: `1,152 / 2` = **576**.

By setting `intermediate_size=576` in our configuration, we ensure that our small model is not just a collection of smaller numbers, but a faithful, principled downscaling of the original architecture, preserving the crucial 3x FFN expansion ratio within its advanced GeGLU structure.

#### Final Configuration Notes

*   **`max_position_embeddings: 1024`**: While the base model supports a native 8192 context length, a smaller context window is appropriate for a model designed for efficiency on potentially shorter-sequence tasks. 1024 is a practical and common choice.
*   **`hidden_activation: "gelu"`**: This is the correct value. The `ModernBERT` implementation in `transformers` is designed to interpret this parameter. It uses the specified activation (`gelu`) within the GeGLU layer's structure. This aligns with the `ModernBertConfig()` default.

---

## Pretraining `ModernBERT-small` for Sentence Embeddings

This section details the pretraining process for `ModernBERT-small`, designed to generate high-quality sentence embeddings. It leverages a robust multi-task, multi-loss training setup to ensure comprehensive semantic understanding.

### Key Features:

*   **Custom Model Architecture:** Initializes a blank `ModernBERT-small` architecture, allowing for training from scratch or continued pretraining.
*   **Multi-Task, Multi-Loss Training:** Employs a diverse set of datasets, each mapped to an appropriate loss function, to build a versatile embedding space.
*   **Modern Loss Functions:** Incorporates advanced loss functions for superior performance and embedding quality.
*   **Comprehensive Evaluation:** Utilizes `TripletEvaluator` and `EmbeddingSimilarityEvaluator` on standard benchmarks for robust validation.
*   **Performance Optimizations:** Configured with `bfloat16` and `flash_attention_2` for efficient training on compatible hardware.

### Datasets Used for Pretraining:

The model is trained on a curated mix of publicly available datasets to ensure broad semantic coverage. Most datasets are used with `CachedMultipleNegativesSymmetricRankingLoss` to promote robust semantic understanding through contrastive learning, while STS Benchmark uses `CoSENTLoss` for regression-based similarity.

1.  **NLI Triplets (`sentence-transformers/all-nli`, `triplet` split):**
    *   **Purpose:** Teaches the model to distinguish between entailment, contradiction, and neutral relationships, crucial for learning semantic similarity and dissimilarity.
    *   **Loss Function:** `CachedMultipleNegativesSymmetricRankingLoss`

2.  **Quora Duplicates Triplets (`sentence-transformers/quora-duplicates`, `triplet` split):**
    *   **Purpose:** Focuses on identifying paraphrases and semantically equivalent questions, enhancing the model's ability to group similar queries.
    *   **Loss Function:** `CachedMultipleNegativesSymmetricRankingLoss`

3.  **Natural Questions Pairs (`sentence-transformers/natural-questions`, `train` split):**
    *   **Purpose:** Trains the model on query-answer pairs, improving its capacity for information retrieval and understanding question-answering contexts.
    *   **Loss Function:** `CachedMultipleNegativesSymmetricRankingLoss`

4.  **STS Benchmark (`sentence-transformers/stsb`, `train` split):**
    *   **Purpose:** A regression task that teaches the model to assign continuous similarity scores to sentence pairs, refining its nuanced understanding of semantic relatedness.
    *   **Loss Function:** `CoSENTLoss`

5.  **Sentence Compression (`sentence-transformers/sentence-compression`, `train` split):**
    *   **Purpose:** Contains pairs of original sentences and their compressed versions, useful for learning semantic equivalence despite structural changes.
    *   **Loss Function:** `CachedMultipleNegativesSymmetricRankingLoss`

6.  **Simple Wiki (`sentence-transformers/simple-wiki`, `train` split):**
    *   **Purpose:** A large corpus of simple English sentences, used to learn general sentence representations and leverage in-batch negatives for contrastive learning.
    *   **Loss Function:** `CachedMultipleNegativesSymmetricRankingLoss`

7.  **AltLex (`sentence-transformers/altlex`, `train` split):**
    *   **Purpose:** Provides pairs of sentences with alternative lexicalizations (different wording for the same meaning), aiding in paraphrase detection.
    *   **Loss Function:** `CachedMultipleNegativesSymmetricRankingLoss`

8.  **COCO Captions (`sentence-transformers/coco-captions`, `train` split):**
    *   **Purpose:** Image captions, offering diverse descriptive language. Used to learn general sentence representations and leverage in-batch negatives.
    *   **Loss Function:** `CachedMultipleNegativesSymmetricRankingLoss`

9.  **Flickr30k Captions (`sentence-transformers/flickr30k-captions`, `train` split):**
    *   **Purpose:** Similar to COCO, provides more image captions for general language understanding and in-batch negative sampling.
    *   **Loss Function:** `CachedMultipleNegativesSymmetricRankingLoss`

10. **Yahoo Answers (`sentence-transformers/yahoo-answers`, `title-question-answer-pair` split):**
    *   **Purpose:** Contains question titles, questions, and answers, useful for learning relationships between different components of a Q&A context.
    *   **Loss Function:** `CachedMultipleNegativesSymmetricRankingLoss`

11. **Stack Exchange Duplicates (`sentence-transformers/stackexchange-duplicates`, `title-title-pair` split):**
    *   **Purpose:** Pairs of duplicate question titles from Stack Exchange, directly training the model on identifying semantically equivalent questions.
    *   **Loss Function:** `CachedMultipleNegativesSymmetricRankingLoss`

### Loss Function Upgrades (Pretraining):

To further improve model performance and output quality during pretraining, the training loss functions were upgraded based on the following rationale:

1.  **`MultipleNegativesRankingLoss` → `MultipleNegativesSymmetricRankingLoss`**
    *   **Reasoning:** The original loss function only trains in one direction (e.g., `query → answer`). The symmetric version adds a second, "backward" loss term (`answer → query`). This creates a more robust and versatile semantic understanding, as the model must learn a reciprocal relationship between sentence pairs, leading to a higher-quality embedding space.

2.  **`CosineSimilarityLoss` → `CoSENTLoss`**
    *   **Reasoning:** `CosineSimilarityLoss` can sometimes produce a compressed range of similarity scores. `CoSENTLoss` is a more modern alternative that directly optimizes the relative ranking of all pairs in a batch. This provides a stronger training signal and typically results in better-calibrated similarity scores that are more spread out and intuitive, improving the model's performance on regression-based similarity tasks.

### Multi-Dataset Batch Sampling Strategy (Pretraining): `PROPORTIONAL` vs. `ROUND_ROBIN`

When training with multiple datasets of varying sizes, the strategy for sampling batches becomes crucial.

*   **`MultiDatasetBatchSamplers.PROPORTIONAL` (Recommended and Used):**
    *   **How it works:** The trainer samples batches from each dataset with a probability proportional to that dataset's size. If Dataset A has 1,000,000 samples and Dataset B has 100,000 samples, Dataset A will be sampled 10 times more often than Dataset B.
    *   **Rationale:** This strategy ensures that larger datasets, which typically contain more unique information, contribute more training steps. This leads to a more balanced overall exposure to the data, preventing smaller datasets from being over-sampled and potentially leading to overfitting on their specific patterns. It generally results in a more robust and general-purpose model, especially with a diverse set of datasets like those used here.

*   **`MultiDatasetBatchSamplers.ROUND_ROBIN` (Alternative):**
    *   **How it works:** The trainer cycles through each dataset, taking one batch from each in sequence (e.g., Dataset A, then B, then C, then A again).
    *   **Consideration:** While it ensures all datasets are seen regularly, it ignores dataset size. A tiny dataset would contribute the same number of batches per cycle as a massive one, potentially leading to disproportionate training time on smaller datasets relative to their total size.

### Evaluation Datasets (Pretraining):

The model's performance is evaluated during pretraining on:

*   **All-NLI Development Set (`sentence-transformers/all-nli`, `dev` split):** Evaluated using `TripletEvaluator`.
*   **STS Benchmark Validation Set (`sentence-transformers/stsb`, `validation` split):** Evaluated using `EmbeddingSimilarityEvaluator` with Cosine Similarity.

**Post-Pretraining Evaluation:**

*   **STS Benchmark Test Set (`sentence-transformers/stsb`, `test` split):** A final, independent evaluation is performed on this dataset after pretraining to provide an unbiased measure of the model's generalization capabilities.

---

## Fine-tuning `ModernBERT-small` for Retrieval Tasks

This script fine-tunes the pretrained `ModernBERT-small` model specifically for information retrieval, aiming to produce high-quality sentence embeddings optimized for search and ranking. It employs a modern multi-dataset training strategy and leverages the powerful `NanoBEIREvaluator` for real-world benchmarking during training.

### Key Features:

*   **Targeted Fine-tuning:** Continues training from the general-purpose `ModernBERT-small` model, specializing it for retrieval.
*   **Retrieval-Focused Datasets:** Utilizes large-scale, real-world datasets designed for information retrieval tasks.
*   **`CachedMultipleNegativesRankingLoss`:** Employs this powerful loss function for in-batch negative training, which is highly effective for learning strong retrieval representations.
*   **`NanoBEIREvaluator`:** Integrates direct benchmarking on subsets of the BEIR (Benchmarking IR) datasets during training. This provides the most accurate and relevant picture of the model's performance on its end-goal task: information retrieval.
*   **Baseline Benchmarking:** Evaluates the model on BEIR datasets *before* fine-tuning to establish a clear performance baseline.
*   **Performance Optimizations:** Continues to use `bfloat16` and `flash_attention_2` for efficient training.

### Datasets Used for Fine-tuning:

The fine-tuning process uses a combination of large-scale retrieval datasets, all formatted for `MultipleNegativesRankingLoss` (triplets or query-positive pairs):

1.  **MS MARCO (`sentence-transformers/msmarco-msmarco-distilbert-base-v3`, `triplet` split):**
    *   **Purpose:** A massive dataset of real-world search queries and relevant passages, crucial for learning robust retrieval capabilities.
    *   **Format:** Triplet (query, positive passage, negative passage).

2.  **GooAQ (`sentence-transformers/gooaq`, `pair` split):**
    *   **Purpose:** Contains question-answer pairs from Google's Natural Questions dataset, useful for improving question-answering retrieval.
    *   **Format:** Pair (question, answer).

3.  **Natural Questions (`sentence-transformers/natural-questions`, `pair` split):**
    *   **Purpose:** Another key dataset for question-answering, providing query-document pairs.
    *   **Format:** Pair (query, answer).

### Loss Function (Fine-tuning):

*   **`CachedMultipleNegativesRankingLoss`:** This loss function is chosen for its effectiveness in retrieval tasks. It optimizes the model to rank relevant documents higher than irrelevant ones within the same batch, leveraging "in-batch negatives" for efficient training. The "Cached" variant further optimizes performance by caching embeddings.

### Multi-Dataset Batch Sampling Strategy (Fine-tuning): `ROUND_ROBIN`

For fine-tuning, the `ROUND_ROBIN` strategy is employed for multi-dataset batch sampling:

*   **`MultiDatasetBatchSamplers.ROUND_ROBIN` (Used):**
    *   **How it works:** The trainer cycles through each dataset, taking one batch from each in sequence (e.g., MSMARCO, then GooAQ, then Natural Questions, then MSMARCO again).
    *   **Rationale:** In this specific fine-tuning context with a smaller number of highly relevant, large datasets, `ROUND_ROBIN` ensures that each critical retrieval dataset contributes an equal number of batches per cycle. This can be beneficial to prevent any single dataset from dominating the training steps, ensuring balanced exposure to different types of retrieval patterns (e.g., long passages vs. short answers).

### Evaluation (Fine-tuning): `NanoBEIREvaluator`

The fine-tuning process uses `NanoBEIREvaluator` for robust and relevant evaluation:

*   **`NanoBEIREvaluator`:** This evaluator directly benchmarks the model on subsets of standard BEIR (Benchmarking IR) datasets. It calculates common retrieval metrics (e.g., nDCG, MRR, Recall) by performing actual retrieval tasks.
*   **Evaluated Datasets:** During fine-tuning, the model is evaluated on:
    *   **MSMARCO**
    *   **NQ (Natural Questions)**
    *   **HotpotQA**
*   **Metric for Best Model:** The `metric_for_best_model` is set to `"eval_NanoHotpotQA_cosine_ndcg@10"`, indicating that the model checkpoint with the best nDCG@10 score on the HotpotQA NanoBEIR subset will be saved as the "best model". This directly aligns the saving strategy with the model's target retrieval performance.
*   **Baseline Evaluation:** The model is evaluated on BEIR datasets *before* training begins to establish a clear performance baseline, allowing for direct comparison of improvement.

---