# README: Domain-Specialized Pre-training (ModernBERT-Student)

Lead Training Engineer Report: **Transitioning from "Cold Start" to "GUIDE-Informed" Pre-training.**

By using the **GUIDE (Guided Weight Initialization)** script, we have fundamentally changed the nature of Step 0. We are no longer training a "blank slate"; we are training a model that has inherited the semantic subspace of a massive Teacher. 

Because your GUIDE script only initializes the **Backbone** (Embeddings and Layer 0), the remaining 11 layers and the **Masked LM Head** will be randomly initialized. This "Warm-Start" at the base of the model provides a massive head start in gradient stability.

### **Refined Training Script: ModernBERT-Student (384-Dim / 12-Layer)**

```python
import os
import torch
import math
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed
)

# 1. ENVIRONMENT SETUP
set_seed(42)

# PATHS: Ensure this points to the OUTPUT_PATH from your initialization script
INIT_MODEL_PATH = "./ModernBERT-small-2/mlm/modernbert-small-init"
TRAIN_FILE = "path/to/your/train.txt" 
OUTPUT_DIR = f"./modernbert-student-pretrained-{datetime.now().strftime('%Y%m%d-%H%M')}"

def run_pretraining():
    # 2. LOAD GUIDE-INITIALIZED MODEL
    print(f"--- Loading GUIDE-Initialized Backbone from {INIT_MODEL_PATH} ---")
    
    # We load via AutoModelForMaskedLM. 
    # It will load the GUIDE weights for the backbone and randomly 
    # initialize the LM head and layers 1-11.
    model = AutoModelForMaskedLM.from_pretrained(
        INIT_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float32, # Maintaining F32 for stability
        attn_implementation="flash_attention_2" # Enabled for speed
    )
    
    tokenizer = AutoTokenizer.from_pretrained(INIT_MODEL_PATH)

    # 3. DATASET PREPARATION (Sentence-Transformers Paragraph Format)
    print("--- Preparing Dataset ---")
    dataset = load_dataset("text", data_files={"train": TRAIN_FILE})

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            return_special_tokens_mask=True,
            return_token_type_ids=False # Strict ModernBERT requirement
        )

    tokenized_datasets = dataset["train"].map(
        tokenize_function, 
        batched=True, 
        remove_columns=["text"],
        num_proc=4
    )

    # 4. COLLATOR (The 30% ModernBERT "Golden Rule")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=True, 
        mlm_probability=0.30
    )

    # 5. TRAINING ARGUMENTS (Optimized for 384-dim Student)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=10,
        
        # Batch Size & Accumulation
        # Since the model is smaller (384 vs 768), you can likely 
        # increase batch size, but we keep it safe for F32 VRAM.
        per_device_train_batch_size=16, 
        gradient_accumulation_steps=2, # Global Batch Size = 32
        
        # Optimizer & Schedule
        learning_rate=8e-4, 
        lr_scheduler_type="cosine",
        warmup_steps=1000, # Reduced warmup because Layer 0/Embeddings are already aligned
        weight_decay=0.01,
        
        # Hardware & Precision
        fp16=False, # Enforce F32
        bf16=False, # Enforce F32
        torch_compile=True, # Recommended by paper for 10% speedup
        gradient_checkpointing=True,
        
        # Logging & Saving
        logging_steps=100,
        save_steps=2500,
        save_total_limit=3,
        report_to="tensorboard",
        logging_first_step=True
    )

    # 6. INITIALIZE TRAINER
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=data_collator,
    )

    # 7. EXECUTION
    print("--- Starting Student Pre-training ---")
    train_result = trainer.train()
    
    # 8. SAVE FINAL BACKBONE
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Perplexity Metric
    metrics = train_result.metrics
    perplexity = math.exp(metrics["train_loss"])
    print(f"Final Student Train Perplexity: {perplexity:.2f}")

if __name__ == "__main__":
    run_pretraining()
```

### **Critical Engineer's Notes for the Student Run**

1.  **The Perplexity Advantage:**
    Because you used **PCA-initialized embeddings**, your starting Loss and Perplexity will be significantly lower than the "Random" runs. A successful GUIDE initialization usually sees the loss drop much faster in the first 500 steps as the model simply learns to connect the new 11 layers to the already-intelligent Layer 0.

2.  **Increased Throughput:**
    Since the model is **384 hidden units** (half the width) and **12 layers** (roughly half the depth of ModernBERT-base), you should expect approximately **2.5x to 3x faster iterations** compared to the base model training.

3.  **VRAM Ceiling:**
    In **Float32**, a 12-layer / 384-dim model is very lean. If your GPU has more than 24GB of VRAM, you can safely increase `per_device_train_batch_size` to **32** or **64** to maximize hardware utilization.

4.  **The "LM Head" Warning:**
    When you start the script, you will see a warning: *"Some weights of ModernBertForMaskedLM were not initialized... [decoder.bias, decoder.weight]"*. 
    **This is expected.** Your GUIDE script built the "Backbone." The Training script is now building the "Head" on top of it. This is exactly what we want.

**Status:** The GUIDE-to-MLM pipeline is now integrated. Proceed with the training run. Control the logs for the first 500 steps to confirm the "Warm Start" effect.


### **The "What": What is this script doing?**
This script is the "Education Phase" of a custom-built language model. It takes a specialized, "Deep & Narrow" model architecture and forces it to learn the unique vocabulary, grammar, and technical nuances of your specific data.

Unlike standard models you download off the shelf, this script performs **Masked Language Modeling (MLM)**. It hides words within your text and challenges the model to guess what they are. By doing this millions of times, the model builds a "mental map" of your specific domain—whether that is medical records, legal briefs, or technical manuals.

---

### **The "Why": Why are we training it this way?**

#### **1. The "Hybrid Start" (Inherited Intelligence)**
We are not starting from a total blank slate. We are using **GUIDE Initialization**.
*   **Why:** Starting a model from random "white noise" takes a massive amount of time and data. Instead, we’ve taken the "DNA" (embeddings and the first layer) of a high-performing teacher model and projected it into this smaller student. 
*   **The Result:** The model starts Day 1 already knowing basic English and general concepts, allowing it to spend the entire training run focusing on the **technical specifics** of your data.

#### **2. The "Deep & Narrow" Architecture**
This model uses a specific configuration: **12 layers deep but only 384 units wide.**
*   **Why:** Most "small" models are shallow (6 layers). By doubling the depth to 12 layers, we give the model more "thinking steps" to process complex sentences. By keeping it narrow (384), we ensure it remains blazingly fast and cheap to run on standard hardware.

#### **3. The 30% Masking Rule (Learning on "Hard Mode")**
Standard models hide 15% of the words during training. This script hides **30%**.
*   **Why:** Modern architectures like ModernBERT are "too smart" for the old 15% standard; they solve the puzzle too easily and stop learning. By hiding nearly a third of the text, we force the model to build a much more robust understanding of context to solve the reconstruction task.

#### **4. Full Numerical Precision (Float32)**
We are running the math in "Full Precision" rather than the "Half Precision" used in standard fine-tuning.
*   **Why:** Training a model's "core" layers is a volatile process. Small mathematical errors in the beginning can snowball and "break" the model (leading to NaN or Inf errors). Full precision provides the mathematical stability needed to ensure the model's weights align perfectly during these critical early steps.

#### **5. Sentence-Transformer Compatibility**
The script treats every line in your data as a standalone paragraph or sentence.
*   **Why:** This model’s ultimate destiny is to become a **Sentence-Transformer** (a model that turns sentences into search vectors). By training it on discrete lines of text now, we are preparing its "brain" for the specific way it will be used later in semantic search.

---

### **The Goal: What do we have when this is finished?**
Once this script completes, you have a **Domain Backbone**. 

It is not yet a search engine, but it is a "Subject Matter Expert" in your data. It understands the jargon, the structure, and the logic of your corpus. This backbone will then be handed off to the next phase, where we teach it how to compare sentences and generate the high-quality vectors used in modern AI search.



Lead Training Engineer Report: **Technical Briefing on Student Pre-training Parameters.**

This configuration is designed to leverage the **ModernBERT "Deep & Narrow"** architecture while maintaining the extreme numerical stability required for a model with a mix of pre-trained (Backbone) and random (Head/Upper Layers) weights.

Here is the breakdown of the settings and the logic behind them:

---

### **1. Throughput & Memory Strategy**
*   **`per_device_train_batch_size=16` & `gradient_accumulation_steps=2`**: 
    *   **The Math:** This creates a **Global Batch Size of 32** ($16 \times 2$). 
    *   **The Rationale:** In `Float32`, weights take up 2x more space than standard training. By using a moderate batch size (16) and accumulating gradients over 2 steps before updating weights, we get the stability of a larger batch without crashing the GPU's memory (VRAM).
*   **`gradient_checkpointing=True`**:
    *   **The Rationale:** This is a memory-saving "hack." It clears the middle-layer activations during the forward pass and re-calculates them during the backward pass. It makes training slightly slower (~20%) but significantly reduces VRAM usage, allowing us to stay in **Float32**.

### **2. The Optimizer & Learning Curve**
*   **`learning_rate=8e-4`**:
    *   **The Rationale:** This is considered a very high learning rate (10x higher than standard fine-tuning). We use it because we are "Cold Starting" 11 out of 12 layers. We need high energy to force those random weights to align with the pre-trained Layer 0 and the new data.
*   **`warmup_steps=1000`**:
    *   **The Rationale:** During the first 1000 steps, the learning rate will gradually climb from 0 to 8e-4. This prevents the "gradient explosion" that happens when a model first sees data with random weights.
*   **`lr_scheduler_type="cosine"`**:
    *   **The Rationale:** After the warmup, the learning rate will follow a smooth curve downward. This helps the model "settle" into the optimal weight values as it nears the end of training.

### **3. Numerical Precision & Modern Performance**
*   **`fp16=False` & `bf16=False`**:
    *   **The Rationale:** This strictly enforces **Float32**. This is your safeguard. By avoiding the rounding errors of 16-bit precision, we ensure that the loss doesn't become `NaN` (Not a Number) during the volatile early phases of training.
*   **`torch_compile=True`**:
    *   **The Rationale:** This is a PyTorch 2.0 feature. It analyzes your model's code and "compiles" it into optimized kernels for your specific GPU. The ModernBERT paper notes this provides a **10% speedup** in throughput.

### **4. Housekeeping & Monitoring**
*   **`save_total_limit=3`**: 
    *   **The Rationale:** Training will save checkpoints every 2500 steps. This setting ensures we only keep the 3 most recent ones, preventing your hard drive from filling up with massive 500MB+ model files.
*   **`report_to="tensorboard"`**:
    *   **The Rationale:** This allows you to visualize the Loss and Perplexity in real-time. You want to see a "smoothly decaying curve."

### **5. The Success Metric: Perplexity**
*   **`perplexity = math.exp(metrics["train_loss"])`**:
    *   **What it is:** Perplexity is a measure of how "surprised" the model is by the text. 
    *   **What to look for:** 
        *   **Step 0:** Perplexity will be ~50,368 (the size of the vocabulary).
        *   **Success:** You want to see this number drop rapidly. For a specialized corpus, a final perplexity between **10 and 40** indicates the model has effectively learned the domain language.


