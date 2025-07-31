# ==============================================================================
#           Configuration for ModernBERT-small
# ==============================================================================
# This configuration defines a 'ModernBERT-small' model, which is a principled
# downscaling of the 'ModernBERT-base' architecture from the original paper.
# The design maintains key architectural ratios to ensure structural consistency
# and performance, resulting in a smaller, faster, and more efficient model.
# ==============================================================================

import os
from transformers import ModernBertConfig, ModernBertModel, AutoTokenizer

# --- 1. Define the Small ModernBERT Architecture ---
from transformers import ModernBertConfig

# Design Rationale:
# This configuration is a principled downscaling of the `ModernBERT-base` architecture.
# It applies the inverse of the scaling rules observed when the authors scaled
# from their `base` to `large` models. This ensures architectural consistency and
# aims to create a smaller, more efficient model that retains the core design
# philosophy of the ModernBERT family.

modernbert_small_config = ModernBertConfig(
    # --- Core Architectural Dimensions ---
    
    hidden_size=384,
    # RATIONALE: A 2x reduction from the ModernBERT-base model's 768 hidden size,
    # establishing a solid foundation for a "small" class model.

    num_hidden_layers=6,
    # RATIONALE: A significant reduction from the base model's 22 layers to create
    # a fast, lightweight model with fewer parameters and lower inference latency.

    num_attention_heads=6,
    # RATIONALE: Derived to maintain a consistent attention head dimension of 64,
    # which is crucial for model stability and performance.
    # CALCULATION: hidden_size / 64 => 384 / 64 = 6.

    # --- Feed-Forward Network (FFN) Configuration ---
    
    intermediate_size=576,
    # RATIONALE: This value maintains the base model's 3.0x FFN expansion ratio,
    # which is critical for the model's representational capacity. For a GeGLU
    # activation, this total expansion is split across two parallel gating layers.
    # CALCULATION: (hidden_size * 3.0) / 2 => (384 * 3.0) / 2 = 576.

    hidden_activation="gelu",
    # RATIONALE: The underlying ModernBERT model code uses this parameter to
    # construct its internal GeGLU activation layer. This is the expected value
    # as per the original model's implementation.

    # --- Other Model Settings ---
    
    max_position_embeddings=1024,
    # RATIONALE: A practical context length for a small, efficient encoder,
    # balancing capability with memory and computational requirements.
)


# --- 2. Create the Blank Model ---
# This initializes a new, blank ModernBERT model from our small configuration.
# The model has the architecture we defined but has not been trained; its
# weights are randomly initialized.
print("Initializing blank ModernBERT model from the configuration...")
model = ModernBertModel(modernbert_small_config)

# Inspect the model architecture.
# Notice the 'score' head at the end, which is a linear layer for classification.
print("\n--- Model Architecture ---")
print(model)

# Check the total number of parameters to see how small it is.
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal number of parameters: {total_params:,}")


# --- 3. Load the Tokenizer ---
# For compatibility, we should use the tokenizer from a pre-trained ModernBERT.
# This ensures our new model understands the token IDs correctly.
print("Loading the tokenizer from 'answerdotai/ModernBERT-base'...")
tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")


# --- 4. Define Save Path and Create Directories ---
# Here we set the specific subdirectory path you requested.
save_directory = "./ModernBERT-small"

# os.makedirs() with exist_ok=True is the robust way to create directories.
# It won't raise an error if the path already exists.
os.makedirs(save_directory, exist_ok=True)
print(f"\nEnsured that save directory exists: {save_directory}")


# --- 5. Save the Model and Tokenizer ---
# We use .save_pretrained() to save all necessary files for both the
# model (config.json, model.safetensors) and the tokenizer.
print(f"Saving model and tokenizer to '{save_directory}'...")

# Save the model's weights and configuration file
model.save_pretrained(save_directory)

# Save the tokenizer's files to the same directory
tokenizer.save_pretrained(save_directory)

print("\n--- Files in the Save Directory ---")
# List the files to confirm everything was saved correctly
for filename in os.listdir(save_directory):
    print(filename)

print(f"\n✅ Success! Your blank small ModernBERT model is saved and ready for training.")
print(f"   You can now point your training script to the path: '{save_directory}'")