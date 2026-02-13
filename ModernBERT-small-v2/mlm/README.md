# ModernBERT-small for MLM

## Guided Weight Initialization (GUIDE) for ModernBERT-Student

## The Objective: "Inherited Foundations"
Standard pre-training usually begins with **random initialization**, where a model spends the first few billion tokens simply learning that the word "apple" is closer to "fruit" than "engine." 

This script bypasses that "infancy" phase. Instead of starting from a blank slate, we use **Guided Weight Initialization (GUIDE)** to surgically shrink a high-performing Teacher model (**ModernBERT-Base**) into a more efficient, "Deep & Narrow" Student model. 

We aren't just copying weights; we are **projecting the teacher's semantic DNA into a smaller subspace.**

---

## 1. The Dimensional Accounting: How the Model was Shrunk
To create a faster, cheaper, yet intelligent encoder, we performed a multi-axis reduction of the architecture. Below is the accounting of the transition from **Base** to **Student**:

| Component | Teacher (Base) | Student (Small) | Change Type | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **Hidden Size ($d_{model}$)** | 768 | **384** | **-50% (Width)** | Reduces vector storage costs by half and speeds up every matrix multiplication. |
| **Num Layers** | 22 | **12** | **-45% (Depth)** | While we reduced depth from the Base 22, we *increased* it relative to standard small models (usually 6 layers) to maintain reasoning power. |
| **Attention Heads** | 12 | **6** | **-50%** | Keeps the **Head Dimension at 64** ($384/6 = 64$), which is the hardware-optimal size for Flash Attention kernels. |
| **Intermediate Size (FFN)** | 1152 | **768** | **-33%** | ModernBERT uses GeGLU. We maintain a **4x expansion ratio** ($384 \times 4 = 1536$ total, split into two 768 blocks) to ensure the model has "room to think." |
| **Vocabulary Size** | 50,368 | **50,368** | **0% (Locked)** | Prevents "semantic fragmentation." Larger vocab means fewer sub-tokens and better hardware alignment. |

---

## 2. The Mechanics of GUIDE Initialization

### A. Subspace Projection (PCA)
We don't just pick the first 384 dimensions of the teacher. We use **Principal Component Analysis (PCA)** on the teacher's embedding matrix. 
*   **The Concept:** PCA identifies the 384 "directions" in the teacher's 768-dimensional space that carry the most information. 
*   **The Result:** We generate a **Projection Matrix ($M$)**. This matrix acts as a bridge, allowing us to map any weight from the 768-space down to our new 384-space while preserving as much semantic "variance" as possible.

### B. Structural Weight Slicing
For the attention and MLP layers, the script performs a "Slice and Project" maneuver:
1.  **Linear Weights:** We use the Projection Matrix ($M$) to compress the input dimensions of the layers.
2.  **Attention Heads:** Since we moved from 12 heads to 6, we take the projected weights and slice them to keep only the first 6 heads. This assumes the teacher’s primary attention patterns are concentrated in the early heads (a common observation in transformer pruning).
3.  **GeGLU Wi/Wo:** Because ModernBERT uses Gated Linear Units, the script carefully splits the weights into "Gate" and "Value" blocks, down-samples them to the new intermediate size (768), and re-joins them. This preserves the gated activation logic.

---

## 3. The "Deep & Narrow" Philosophy
Most small models are "Shallow & Wide" (e.g., 6 layers, 768 hidden). We have chosen **Deep & Narrow (12 layers, 384 hidden).**

*   **Why Depth?** Semantic search and complex reasoning require "multi-hop" processing. Each layer is a step of logic. 12 layers allow the model to refine its understanding of a sentence much more than 6 layers can, even if the individual layers are "narrower."
*   **Why Narrow?** A hidden size of 384 is the "sweet spot" for modern semantic search. It is small enough to be blazingly fast on CPU and GPU, and it results in smaller vector database indexes, saving money on infrastructure.

---

## 4. Why Layer 0?
The script explicitly initializes **Layer 0** of the student using the teacher's Layer 0. 
*   **The Logic:** In a transformer, the first layer is responsible for "cleaning up" the raw embeddings and identifying basic syntax. By inheriting a projected version of the teacher's Layer 0, the student starts its training already knowing how to parse the language. 
*   **The Rest:** Layers 1–11 remain randomly initialized. This "Warm-Start" at the bottom of the model stabilizes the gradients during the early phases of MLM training, preventing the "NaN" losses common in cold-starts.

---

## 5. Summary of the Resulting Model
The output of this script is a **Hybrid Initialize Model**:
1.  **Embeddings:** Semantically dense, 384-dim PCA-compressed versions of answerdotai-Base.
2.  **Layer 0:** A projected "Logic Gate" inherited from the teacher.
3.  **Layers 1-11:** Blank slate, ready to be filled with domain-specific knowledge during your MLM pre-training.

**This is the most efficient starting point for a specialized Sentence-Transformer.**

### Leave the vocabulary as-is (50,368)

As the Lead Training Engineer, here is the technical justification for why changing the vocabulary now would jeopardize your **GUIDE (Guided Weight Initialization)** strategy:

### 1. The GUIDE Strategy depends on Teacher Alignment
Your script uses `answerdotai/modernbert-embed-base` as the Teacher. 
*   **The Projection Matrix ($M$):** You are calculating a PCA projection from the teacher's embedding space (768) to your student's space (384). 
*   **The Index Mapping:** PCA is performed across the `vocab_size` dimension. If you were to shrink the student vocabulary to, say, 30,000, you would have a mismatch. You would have to decide which 20,368 tokens to delete from answerdotai's vocabulary. If you delete the wrong ones, your student becomes "illiterate" to terms the teacher considers vital.
*   **The Result:** By keeping the vocab size identical, you ensure a **1:1 identity mapping** for every token. Every specialized term answerdotai knows, your student now knows (albeit in a compressed 384-dimensional vector).

### 2. The MLP and Attention Projections
In your code:
```python
t_wqkv_proj = np.dot(t_wqkv, M) 
```
This math relies on the Projection Matrix ($M$) derived from the embeddings. Because the embeddings are the "anchor" for the hidden states, the vocabulary size dictates the "shape" of the data flowing through the layers. Keeping the vocabulary at 50,368 ensures that the weights you project from Teacher Layer 0 into Student Layer 0 are mathematically consistent with the compressed embedding space.

### 3. "Tie Word Embeddings" is your Safety Net
In your configuration:
```python
tie_word_embeddings=True
```
Because you are tying the embeddings, you aren't actually "paying" for the large vocabulary twice. The model uses the same memory for the input (token -> vector) and the output (vector -> token prediction). This is the standard "Modern" way to have a high-resolution vocabulary without doubling the parameter count of the embedding layers.

### 4. Semantic Preservation in "Deep & Narrow"
You've chosen a **12-layer** student. This is a "smart" depth. 
*   If you had a 512-token sequence and a **small** vocab, that sequence might be broken into 800 sub-tokens, forcing you to truncate half your data.
*   With the **50,368** vocab, that same 512-token sequence likely stays around 550-600 tokens. 
*   Your 12-layer model will be much more effective at building sentence embeddings if it starts with "high-quality bricks" (tokens that represent whole words/concepts).

### **Final Script Sanity Check**
Your implementation of the GUIDE logic is excellent, particularly the handling of the **GeGLU split**:
```python
t_wi_gate, t_wi_val = np.split(t_wi_proj, 2, axis=0)
s_wi_gate = t_wi_gate[:768, :]
s_wi_val = t_wi_val[:768, :]
```
This is the correct way to down-sample a GeGLU layer. If you were to change the vocabulary now, you'd have to rewrite the PCA logic and risk losing the "semantic DNA" you are trying to inherit from the answerdotai teacher.

**Engineer's Verdict:** Keep the `vocab_size=50368`. It is the "glue" that holds your GUIDE initialization and your specialized MLM pre-training together. Proceed with the initialization.
