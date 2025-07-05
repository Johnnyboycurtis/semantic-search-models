# build_and_save_small_modernbert.py

import os
from transformers import ModernBertConfig, ModernBertModel, AutoTokenizer

# --- 1. Define the Small ModernBERT Architecture ---
# This configuration creates a much smaller model than the ModernBERT-base.
# We are reducing the key parameters that determine the model's size.
print("Defining the small ModernBERT configuration...")
small_modernbert_config = ModernBertConfig(
    hidden_size=256,                 # A common dimension for small embedding models
    num_hidden_layers=6,               # Significantly fewer layers than the base's 22
    num_attention_heads=4,             # Must be a divisor of hidden_size
    intermediate_size=1024,            # Typically 4 * hidden_size
    max_position_embeddings=512,       # Max sequence length for the model
)

# --- 2. Create the Blank Model ---
# This initializes a new, blank ModernBERT model from our small configuration.
# The model has the architecture we defined but has not been trained; its
# weights are randomly initialized.
print("Initializing blank ModernBERT model from the configuration...")
model = ModernBertModel(small_modernbert_config)

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