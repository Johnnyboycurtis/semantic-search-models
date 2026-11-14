# Optimization Schedules for Partially Initialized Embedding Models: Freezing Dynamics in Parameter-Space Transfer and Distillation

## Theoretical Foundations and Dynamics of Partial Parameter Transfer

Initializing deep transformer architectures through partial parameter transfer introduces structural optimization asymmetries$^1$. In the Guided Initialization and Distillation of Embeddings (GUIDE) framework, a compact student model inherits structural priors from an uncompressed teacher by projecting the teacher token embedding matrix $E_T \in \mathbb{R}^{V \times d_T}$ into a reduced subspace $E_S \in \mathbb{R}^{V \times d_S}$ via principal component analysis (PCA), while slicing and transforming initial transformer layers using the projection matrix $M \in \mathbb{R}^{d_T \times d_S}$ $^1$. When the remaining intermediate blocks (Layers 1–11 in a 12-layer architecture) and prediction heads are initialized randomly, the network exists in a bifurcated state: the lower representations occupy an informative, low-rank manifold, whereas the subsequent transformations generate isotropic noise$^1$.

During the initial iterations of Masked Language Modeling (MLM), an uncalibrated vocabulary projection head operating over a vocabulary size of $V = 50,368$ produces an initial cross-entropy loss approaching $\ln(50368) \approx 10.83$. Backpropagating this error through eleven randomly parameterized layers concentrates stochastic error gradients onto the lower layers$^2$. Theoretical analyses of two-timescale dynamics and feature distortion demonstrate that when a pre-trained feature extractor is coupled with an unaligned linear or deep head, unconstrained gradient descent immediately updates the feature extractor along directions spanning the batch data rather than preserving the global manifold$^2$.

This phenomenon, known as feature distortion, occurs because early gradient vectors do not represent structural semantic corrections; instead, they reflect high-variance attempts by the optimizer to counteract upper-layer noise$^2$. In standard adaptive moment estimation (AdamW), large gradient magnitudes distort the first and second moment vectors ($m_t$ and $v_t$), rapidly overwriting the orthogonal subspace defined by the PCA projection matrix $M^1$. Consequently, the parametric bridge connecting the student's initial state to the teacher's geometry collapses before intermediate layers develop coherent self-attention routing$^1$.

The warm-start dilemma in partially initialized transformers therefore reflects a tension between optimization stability and representational plasticity$^1$. If the projected layers are updated freely from step zero, the benefits of parameter transfer are largely erased by destructive early updates$^2$. Conversely, if the transferred layers remain permanently frozen, the architecture cannot adapt to the truncated latent dimensionality, limiting downstream task performance$^1$.

## Comparative Analysis of Pre-training Regimes

The literature on neural network pruning, model growth, and transfer learning presents four distinct strategies for managing parameter updates across partially initialized networks$^2$.

| Strategy | Optimization Dynamics | Risk Profile | Empirical Benchmark Utility |
| :--- | :--- | :--- | :--- |
| **Option A: Complete Hard Freeze** | Gradient flow is truncated above Layer 0; $E_S$ and Layer 0 remain static across all pre-training steps$^7$. | Induces representational rigidity; prevents compensation for discarded PCA variance$^1$. | Underperforms on downstream classification and retrieval tasks requiring corpus adaptation$^2$. |
| **Option B: Unconstrained Step-0 Training** | A uniform learning rate is applied across all parameters from the initial optimization step$^2$. | Causes severe feature distortion; destroys low-rank teacher geometry during early loss spikes$^2$. | Matches or falls below random initialization baselines; negates transfer acceleration$^1$. |
| **Option C: Two-Stage Gradual Unfreezing (LP-FT)** | Lower layers are frozen while upper layers align, followed by full parameter unfreezing$^2$. | Introduces scheduling complexity and potential gradient shock at the unfreezing boundary$^8$. | Consistently maximizes out-of-distribution robustness and retention of transferred priors$^2$. |
| **Option D: Layer-wise Learning Rate Decay (LLRD)** | Multiplicative learning rate scaling $\eta_l = \eta \cdot \gamma^{L-l}$ establishes a continuous learning rate gradient$^{10}$. | Bottom layers remain exposed to early noisy updates if the warmup phase is brief$^2$. | Highly effective for pre-trained backbones; partially mitigates gradient shock in mixed setups$^9$. |

### Complete Hard Freezing
Holding transferred parameters static isolates the inherited subspace from gradient corruption$^4$. In low-compute language modeling, initializing word embeddings on in-domain corpora and freezing them preserves representations of rare tokens that would otherwise suffer from under-sampling$^7$.

However, permanent hard freezing imposes a rigid geometric bottleneck$^1$. Truncating a 768-dimensional embedding to 384 dimensions via PCA discards secondary variance components$^1$. If the embedding matrix $E_S$ cannot adjust its coordinate axes to compensate for omitted singular vectors, the upper layers must expend capacity resolving residual projection errors, capping downstream language modeling performance$^1$.

### Unconstrained Step-0 Optimization
Subjecting projected parameters to full optimization from the outset induces rapid anchor drift$^1$. In the original GUIDE formulation, Google Research observed that naive initialization methods—such as the "first-$N$ selection" proposed in prior model growth heuristics—performed poorly due to optimization instabilities when integrated into full transformer backbones$^1$.

When gradient backpropagation is unrestricted, the initial update trajectory is dominated by the cross-entropy magnitude of the uninitialized prediction head$^2$. Theoretical bounds establish that out-of-distribution error remains elevated throughout gradient descent if the initial steps distort the underlying representation manifold$^2$. The model effectively degenerates toward a pseudo-random initialization, discarding the computational advantage provided by the teacher's transferred covariance structure$^1$.

### Gradual Unfreezing and Head-First Warming
The Linear Probing then Fine-Tuning (LP-FT) framework demonstrates both theoretically and empirically that sequentially optimizing uninitialized layers before updating transferred representations prevents feature distortion$^2$. In the first stage, the feature base is held static while randomly initialized parameters minimize task loss to convergence or near-stability$^2$. In the second stage, the entire network is fine-tuned end-to-end with a conservative learning rate$^2$.

Applying this paradigm to deep-and-narrow transformer architectures enables Layers 1–11 and the MLM head to establish valid syntactic routing and unigram output baselines against the static projected coordinate system of $E_S$ and Layer 0$^{15}$. Once the MLM loss stabilizes from its initial spike ($\approx 10.8$) to a standard training trajectory ($\approx 3.0-3.5$), unfreezing the bottom layers permits holistic adaptation without subspace erasure$^2$.

### Discriminative Learning Rates and LLRD
Layer-wise Learning Rate Decay assigns parameter updates according to layer depth:
$$\eta_l = \eta_{base} \cdot \gamma^{L-l}$$
where $\gamma \in (0,1)$ denotes the decay factor, $L$ represents total depth, and $l$ denotes the current layer index$^{10}$. In model growth frameworks such as bert2BERT and LiGO, scaling down the learning rates of warm-started components ensures stable optimization during continuous pre-training$^9$.

Setting $\gamma \approx 0.8-0.85$ applies an order-of-magnitude lower learning rate to the embedding table and Layer 0 relative to the top transformer blocks$^{10}$. While LLRD dampens the amplitude of disruptive updates, it does not completely eliminate adverse early updates during the first few hundred steps unless paired with an initial frozen phase$^2$.

## Empirical Benchmarks and Ablation Studies

Empirical evidence across transformer downscaling, pruning, and distillation demonstrates the direct performance benefits of protecting transferred representations$^1$.

In the GUIDE framework evaluated by Google Research, parameter-space initialization was compared directly against traditional knowledge distillation and random initialization across large transformer models$^1$. Combining parameter initialization with prediction-space distillation yielded compound improvements, illustrating that parameter transfer and output distillation target complementary phases of the optimization process$^1$.

| Model / Initialization Variant | Language Model Perplexity (↓) | Teacher-Student Gap Reduction (%) |
| :--- | :--- | :--- |
| **Teacher Baseline (4.2B parameters)** | **9.621 ± 0.004** | — |
| Random Initialization Student (1B parameters) | 13.382 ± 0.012 | — |
| Knowledge Distillation Alone (KD) | 12.636 ± 0.011 | 19.86% |
| GUIDE Initialization Alone | 12.438 ± 0.011 | 25.11% |
| **GUIDE + Knowledge Distillation Combined** | **11.813 ± 0.010** | **41.73%** |

The GUIDE evaluation confirms that preserving structural alignment at initialization reduces the optimization burden during subsequent training phases$^1$. Rather than forcing the student to learn both internal coordinate bases and token distributions simultaneously, the model begins training from an aligned representational manifold$^1$.

Complementary evidence from TinyBERT illustrates the necessity of explicit embedding-level alignment during distillation$^{18}$. In comprehensive component ablations on the General Language Understanding Evaluation (GLUE) benchmark, removing embedding-layer distillation resulted in significant drops across syntactic and semantic tasks$^{18}$.

| TinyBERT Configuration | GLUE Average Score | MNLI (m/mm) | QNLI | SST-2 |
| :--- | :--- | :--- | :--- | :--- |
| **TinyBERT Full Architecture** | **77.0** | **79.7 / 79.9** | **87.7** | **92.6** |
| TinyBERT w/o Embedding Distillation ($\mathcal{L}_{embd}$) | 74.1 | 77.8 / 78.1 | 85.0 | 82.3 |
| TinyBERT w/o Prediction Layer Distillation ($\mathcal{L}_{pred}$) | 73.5 | 77.1 / 77.3 | 84.3 | 80.5 |
| TinyBERT w/o Attention Matrix Distillation ($\mathcal{L}_{attn}$) | 73.0 | 76.8 / 76.2 | 84.0 | 88.3 |

The drop from 77.0 to 74.1 demonstrates that regularizing the embedding manifold against the teacher preserves representational fidelity far more effectively than leaving the embeddings unconstrained without guidance$^{18}$. In architectures where hidden dimensions differ, intermediate projections bridge the vector spaces, preventing premature representational drift$^{18}$.

## Downstream Impact on Semantic Representation and Retrieval Quality

The training regime directly governs the downstream retrieval effectiveness of bi-encoder architectures evaluated on benchmarks such as BEIR and MTEB$^{15}$. Dense semantic retrieval models rely on the fine-grained inner product geometry of their output vectors$^{22}$. If token embeddings are frozen indefinitely during domain pre-training, the model retains the global lexical topology of the teacher but fails to adjust to domain-specific co-occurrence shifts and the narrower 384-dimensional latent channel$^1$.

Empirical results from token distillation and modular adaptation confirm that input embedding adaptability is essential for fine-grained retrieval$^{13}$. Freezing input token embeddings limits the capacity of intermediate self-attention heads to optimize attention dispersion across specialized terminology$^7$.

Conversely, premature or unconstrained training from step zero causes representational collapse, wherein the isotropic geometry preserved by PCA projection degrades into degenerate, highly anisotropic clusters$^3$. In dense retrieval pipelines, anisotropic representations degrade Maximum Inner Product Search (MIPS) by collapsing cosine similarity dynamic ranges into narrow, high-correlation domains$^{22}$. Gradual unfreezing balances this tradeoff by anchoring the global coordinate system before granting parameters the plasticity required to maximize downstream token discrimination$^2$.

## Knowledge Distillation Dynamics for Dense Bi-Encoders

When transitioning from Stage A (Domain MLM Pre-training) to Stage B (Dense Knowledge Distillation), the optimization objective shifts from token-level classification to dense vector mapping$^{15}$. In bi-encoder pipelines, such as those standard in Sentence-Transformers, the student is optimized to match the output representations of an advanced teacher (e.g., gte-modernbert-base) across text pairs or triplets$^{15}$.

### End-to-End Distillation versus Hard Freezing
Industrial distillation pipelines for dense semantic search typically train the entire student backbone end-to-end rather than holding lower layers fixed$^{15}$. This practice stems from architectural considerations: because the student's hidden dimension ($d=384$) differs from that of the teacher ($d=768$), the final sentence embedding (derived via mean pooling over the sequence) must compress contextual semantics into half the vector bandwidth$^{15}$.

If the token embedding table is held static during dense distillation, the upper layers must compensate for any representational discrepancies in the input tokens, introducing optimization bottlenecks$^{18}$. End-to-end training using Mean Squared Error (MSELoss) on pooled vectors or margin alignment (MarginMSELoss) across pairs allows the full parameter volume to accommodate the reduced dimensionality$^{15}$.

### Auxiliary Embedding Alignment Objectives
Rather than enforcing rigid parameter freezes during distillation, established frameworks introduce auxiliary intermediate alignment objectives$^{18}$. As pioneered in FitNets and formalized for transformers in TinyBERT, multi-layer distillation incorporates an explicit embedding alignment loss:
$$\mathcal{L}_{embd} = MSE(E_S W_e, E_T)$$
where $W_e \in \mathbb{R}^{d_S \times d_T}$ represents a learnable linear transformation mapping the student's embedding space into the teacher's coordinate frame$^{18}$. In the GUIDE framework, this linear mapping $W_e$ is replaced by the transpose of the PCA projection matrix $M^T \in \mathbb{R}^{d_S \times d_T}$, directly aligning parameter spaces without adding parameter overhead$^1$.

When combined with output distillation, this structural alignment anchors the bottom representations while enabling end-to-end task refinement$^1$.

## Documented Failure Modes and Optimization Pitfalls

Empirical studies in transformer downscaling, pruning, and parameter transfer identify three primary optimization failure modes$^1$.

### Subspace Overwriting and Anchor Drift
Allowing full gradient backpropagation into projected parameters while upper layers remain unaligned results in rapid subspace overwriting$^2$. Because randomly initialized Layers 1–11 generate quasi-random vectors, early gradient updates to $E_S$ exhibit high directional entropy$^2$.

The adaptive learning rate in AdamW scales these updates by the inverse of past gradient variances, causing rapid parameter changes that overwrite the principal component axes inherited from the teacher$^1$. By step 1,000, the token embeddings diverge from the teacher's geometric manifold, degrading the quality of the warm start$^2$.

### Manifold Inflexibility and Capacity Stagnation
Conversely, maintaining a hard freeze on $E_S$ and Layer 0 throughout the entire training duration locks the architecture into an incomplete subspace$^1$. Truncating the hidden dimension from 768 to 384 discards information contained in lower-order eigenvalues$^1$.

Under an indefinite freeze, the model cannot adjust its embedding coordinates to capture domain-specific vocabulary nuances or compensate for missing teacher dimensions$^7$. This creates an information bottleneck at Layer 1, yielding suboptimal downstream retrieval performance on specialized corpora$^{13}$.

### Gradient Shock at the Unfreezing Boundary
When transitioning a parameter block from a frozen state (requires_grad = False) to an active training state (requires_grad = True), naive training schedules often encounter optimization instability at the transition boundary$^8$. If the optimizer maintains stale state estimates or if the transition uses an unadjusted learning rate without a secondary warmup, the newly unfrozen layers experience large initial updates$^{12}$.

This sudden injection of parameter variance destabilizes the convergence achieved by the upper layers, resulting in transient loss spikes and localized forgetting$^8$.

## Practical Training Recommendations for ModernBERT-Small

To maximize representational stability and domain adaptation in a 12-layer, 384-dimensional ModernBERT student model initialized via GUIDE, training should be structured as an integrated multi-phase curriculum$^1$.

The optimization lifecycle comprises three operational phases across two primary training stages:

1.  **Phase A1 (Alignment Warmup, Steps 0–5,000):** Token embeddings $E_S$ and Layer 0 are completely frozen (requires_grad = False), while Layers 1–11 and the output MLM projection head are fully trainable$^1$. Training begins with a base learning rate of $\eta_{top} = 5 \times 10^{-4}$ using AdamW, incorporating a linear warmup over the first 2,000 steps$^{17}$. This phase absorbs the initial cross-entropy error (loss $\approx 10.8$) and allows intermediate self-attention heads to align with the fixed 384-dimensional coordinate bridge provided by Layer 0$^1$. Phase A1 concludes once the MLM loss stabilizes within a standard convergence range ($\approx 3.0-3.5$)$^{15}$.
2.  **Phase A2 (Joint Subspace Adaptation, Steps 5,000 to End of MLM):** The token embedding table $E_S$ and Layer 0 are unfrozen (requires_grad = True), and Layer-wise Learning Rate Decay is applied with a decay factor $\gamma = 0.82$ $^8$. The effective learning rate for each layer scales relative to a base rate $\eta_{base} = 3 \times 10^{-4}$:
    $$\eta_{Layer \, l} = \eta_{base} \cdot (0.82)^{12-l}$$
    $$\eta_{E_S} = \eta_{base} \cdot (0.82)^{13} \approx 2.5 \times 10^{-5}$$
    To prevent gradient shock at the transition boundary, an internal linear warmup is applied to $E_S$ and Layer 0 over steps 5,000 to 7,000, ramping their learning rates from zero to their designated levels, followed by a joint cosine decay schedule across all parameters$^8$. This phase enables fine-grained parameter updates that adapt token representations to domain-specific syntax without distorting the global manifold$^2$.
3.  **Phase B (Dense Knowledge Distillation):** The model transitions to bi-encoder specialization using gte-modernbert-base as the teacher$^{15}$. All layers remain fully trainable (requires_grad = True), optimized using AdamW with a base learning rate $\eta = 1 \times 10^{-4}$, a 1,000-step linear warmup, and cosine decay to $1 \times 10^{-6}$ $^{15}$. Sentence representations are aligned using Mean Squared Error (MSELoss) on mean-pooled vectors or Margin Mean Squared Error (MarginMSELoss) across passage retrieval triplets$^{15}$. Full-backbone training during this stage ensures that all parameters adapt holistically to the compressed 384-dimensional semantic space$^{15}$.

## Detailed Hyperparameter Specifications

The structured schedule below outlines the complete hyperparameter configuration across each training phase for the 12-layer, 384-dimensional architecture:

| Hyperparameter / Component | Phase A1: Alignment Warmup | Phase A2: Joint MLM Pre-training | Phase B: Dense Distillation |
| :--- | :--- | :--- | :--- |
| **Objective Function** | Masked Language Modeling | Masked Language Modeling | Mean Squared Error / Margin MSE$^{15}$ |
| **Token Embeddings ($E_S$)** | Frozen (requires_grad = False)$^4$ | Trainable (requires_grad = True)$^2$ | Trainable (requires_grad = True)$^{15}$ |
| **Layer 0 Weights ($W_{S,0}$)** | Frozen (requires_grad = False)$^4$ | Trainable (requires_grad = True)$^2$ | Trainable (requires_grad = True)$^{15}$ |
| **Layers 1–11 & Projection Head** | Trainable (requires_grad = True)$^8$ | Trainable (requires_grad = True)$^8$ | Trainable (requires_grad = True)$^{15}$ |
| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.98, \epsilon=10^{-6}$)$^1$ | AdamW ($\beta_1=0.9, \beta_2=0.98, \epsilon=10^{-6}$)$^1$ | AdamW ($\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$)$^{15}$ |
| **Peak Learning Rate (Layers 1–11)** | **$5.0 \times 10^{-4}$**$^{27}$ | **$3.0 \times 10^{-4}$**$^1$ | **$1.0 \times 10^{-4}$**$^{15}$ |
| **Peak Learning Rate ($E_S$, Layer 0)** | **0.0** (Inactive) | **$2.5 \times 10^{-5}$** (via LLRD $\gamma=0.82$)$^{10}$ | **$1.0 \times 10^{-4}$** (Uniform or $\gamma=0.90$)$^{15}$ |
| **Warmup Duration** | 2,000 steps (Linear)$^{17}$ | 2,000 steps secondary linear warmup on $E_S$$^8$ | 1,000 steps (Linear)$^{27}$ |
| **Learning Rate Schedule** | Linear warmup to constant | Cosine decay to $1.0 \times 10^{-5}$ $^{16, 28}$ | Cosine decay to $1.0 \times 10^{-6}$ $^{28}$ |
| **Weight Decay** | **0.01** (Decoupled)$^{17}$ | **0.01** (Decoupled)$^{17}$ | **0.01**$^{17}$ |
| **Gradient Clipping ($\ell_2$ Norm)** | **1.0**$^1$ | **1.0**$^1$ | **1.0**$^1$ |
| **Per-Device Batch Size** | Architecture-dependent (e.g., 64–128)$^{15}$ | Architecture-dependent (e.g., 64–128)$^{15}$ | 64–128$^{15}$ |

Implementing this structured optimization schedule resolves the warm-start dilemma$^1$. By protecting the PCA-projected parameters during early high-variance MLM steps and reintroducing gradient flow through layer-decayed learning rates, the student model retains the structural geometry inherited from the teacher while acquiring the domain adaptation necessary for high-accuracy semantic retrieval$^1$.

***

### Works cited

1.  Guided Initialization and Distillation of Embeddings - arXiv, https://arxiv.org/pdf/2510.06502
2.  Fine-Tuning can Distort Pretrained Features and Underperform Out, https://arxiv.org/pdf/2202.10054
3.  Heimr 570M: a local model that stays fast as the context grows, https://sevren.ai/blog/heimr-570m
4.  Two-Timescale Fine-tuning Provably Learns New Features ... - arXiv, https://arxiv.org/pdf/2609.34667
5.  TransHands: Repurposing Human Pose Encoders as Hand Pose, https://openreview.net/pdf?id=kJKITrryep
6.  EARLY PERIOD OF TRAINING IMPACTS ADAPTATION FOR OUT, https://openreview.net/pdf?id=qDFpNXnuYK
7.  Improving Low Compute Language Modeling with In-Domain, https://aclanthology.org/2020.emnlp-main.696.pdf
8.  arXiv:2403.13341v2 [cs.CV] 3 Jun 2024, https://arxiv.org/pdf/2403.13341
9.  bert2BERT: Towards Reusable Pretrained Language Models, https://aclanthology.org/2022.acl-long.151.pdf
10. Masked Autoencoders are Secretly Efficient Learners - IEEE Xplore, https://ieeexplore.ieee.org/document/10677918/
11. Automatically Freezing Model Blocks to Accelerate Fine-tuning - arXiv, https://arxiv.org/html/2102.01386v2
12. generation/run-2026-08-07-opus-5-iclrtopics-v1/19 ... - Hugging Face, https://huggingface.co/datasets/humainlab/llm-research-index/blob/3600cb414dca59b174aecfb5971ca62fadc837a9/generation/run-2026-08-07-opus-5-iclrtopics-v1/19-optimization/idea_0018.md
13. TOKEN DISTILLATION: ATTENTION-AWARE INPUT EMBEDDINGS, https://proceedings.iclr.cc/paper_files/paper/2026/file/e3cd4c3748274cfde024873d4ae91600-Paper-Conference.pdf
14. Learning Dynamics in RL Post-Training for Language Models - arXiv, https://arxiv.org/pdf/2601.04670
15. johnnyboycurtis/ModernBERT-small-v2 - Hugging Face, https://huggingface.co/johnnyboycurtis/ModernBERT-small-v2
16. MobileLLM: Optimizing Sub-billion Parameter Language Models for, https://openreview.net/pdf?id=EIGbXbxcUQ
17. learning to grow pretrained models for efficient transformer training, https://cap.csail.mit.edu/sites/default/files/research-pdfs/LEARNING%20TO%20GROW%20PRETRAINED%20MODELS%20FOR%20EFFICIENT%20TRANSFORMER%20TRAINING.pdf
18. TINYBERT: DISTILLING BERT FOR NATURAL LAN - OpenReview, https://openreview.net/pdf?id=rJx0Q6EFPB
19. TinyBERT: Distilling BERT for Natural Language Understanding, https://aclanthology.org/2020.findings-emnlp.372.pdf
20. Layer-Level Knowledge Distillation for Deep Neural Network Learning, https://www.mdpi.com/2076-3417/9/10/1966
21. 14 Powerful Techniques Defining the Evolution of Embedding, https://www.analyticsvidhya.com/blog/2025/04/evolution-of-embeddings/
22. Losses - Sentence Transformers documentation, https://sbert.net/docs/package_reference/sentence_transformer/losses.html
23. distillation for information retrieval - OpenReview, https://openreview.net/pdf?id=-aEuKX6zQKmr
24. Token Distillation: Attention-Aware Input Embeddings for New Tokens, https://arxiv.org/html/2505.20133v2
25. arXiv:2303.13220v1 [cs.IR] 23 Mar 2023, https://arxiv.org/pdf/2303.13220
26. A Gift From Knowledge Distillation: Fast Optimization, Network, https://openaccess.thecvf.com/content_cvpr_2017/papers/Yim_A_Gift_From_CVPR_2017_paper.pdf
27. Building Mixture-of-Experts from LLaMA with Continual Pre-Training, https://aclanthology.org/2024.emnlp-main.890.pdf
28. sheared llama: accelerating language model pre-training via ... - arXiv, https://arxiv.org/pdf/2310.06694
29. sentence-transformers/msmarco-bert-base-dot-v5 - Hugging Face, https://huggingface.co/sentence-transformers/msmarco-bert-base-dot-v5