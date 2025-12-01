import logging
import os
import shutil
from datasets import load_dataset
from transformers import ModernBertConfig, ModernBertModel, AutoTokenizer
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.evaluation import NanoBEIREvaluator


# 1. Setup Logging
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)

# ==========================================
# PART 1: Initialize Custom Model from Scratch
# ==========================================

logger.info("Initializing custom ModernBERT-Small configuration...")

# We borrow the tokenizer from the base model so we have a valid vocabulary
# ModernBERT uses a specific vocab size (usually 50368 or 50280 depending on implementation)
# We load the base tokenizer to ensure compatibility.
base_model_id = "nomic-ai/modernbert-embed-base"
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# Define your Custom Configuration
from transformers import ModernBertConfig

# Previous Small Config (for reference):
# hidden_size=384, layers=6, intermediate=576 (1.5x expansion due to split)

config = ModernBertConfig(
    # --- WIDTH (Unchanged) ---
    # Keeping this at 384 ensures your vector storage costs remain identical.
    hidden_size=384,
    
    # --- DEPTH (Doubled) ---
    # We move from 6 layers to 12. 
    # This matches the depth of standard DistilBERT/MiniLM models, allowing
    # for significantly more complex semantic reasoning (multi-hop logic).
    num_hidden_layers=12,
    
    # --- HEADS (Unchanged) ---
    # 384 / 64 = 6. This must stay 6 to preserve the 64-dim head size.
    num_attention_heads=6,

    # --- FFN CAPACITY (Principled Increase) ---
    # In your previous Small model, you used intermediate_size=576.
    # While mathematically valid for GeGLU (1.5x * 2 = 3.0x total), 
    # it is very thin for a 12-layer model. 
    #
    # Standard BERT/ModernBERT scaling usually targets a 4.0x total expansion.
    # Calculation: 
    #   Target Expansion: 384 * 4.0 = 1536 total units.
    #   GeGLU Split: 1536 / 2 = 768.
    #
    # We set this to 768 or 1152. 
    # 1152 preserves the *exact* ratio of the Base model (Base is 1152 for 768 input).
    # Since we are 384 input (half of base), 1152 is actually a massive 6x expansion (3x split).
    #
    # RECOMMENDATION: 768.
    # This gives you a 4.0x total expansion (standard for deep models),
    # ensuring the deeper layers have enough "memory" without bloating compute.
    intermediate_size=768, 

    # --- Standard ModernBERT Settings ---
    hidden_activation="gelu",
    max_position_embeddings=1024, # Keep 1024 or bump to 2048 if your RAG chunks are larger
    vocab_size=50368, # Ensure this matches your tokenizer
    
    # Critical for training stability on deeper models
    attn_implementation="flash_attention_2", 
)
# Initialize the model with random weights based on the config
hf_model = ModernBertModel(config).to("cuda")
print(hf_model)

# Save this "skeleton" model to disk so SentenceTransformer can load it correctly
custom_model_path = "./ModernBERT-small-2/modernbert-small-init"
if os.path.exists(custom_model_path):
    shutil.rmtree(custom_model_path)
    
logger.info(f"Saving initialized model to {custom_model_path}...")
hf_model.save_pretrained(custom_model_path)
tokenizer.save_pretrained(custom_model_path)

# Now load it as a SentenceTransformer
# This automatically adds the necessary Pooling layer on top
logger.info("Wrapping in SentenceTransformer...")
model = SentenceTransformer(custom_model_path)

