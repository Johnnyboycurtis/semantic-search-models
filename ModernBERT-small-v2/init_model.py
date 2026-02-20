import logging
import os
import shutil
import torch
import numpy as np
from sklearn.decomposition import PCA
from transformers import ModernBertConfig, ModernBertModel, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

# 1. Setup Logging
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)

# ==========================================
# PART 1: Configuration
# ==========================================

TEACHER_ID = "nomic-ai/modernbert-embed-base"
OUTPUT_PATH = "./ModernBERT-small-2/modernbert-small-init"

# Your Custom "Deep & Narrow" Config
student_config = ModernBertConfig(
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
    max_position_embeddings=1024,  # Keep 1024 or bump to 2048 if your RAG chunks are larger
    vocab_size=50368,  # Ensure this matches your tokenizer
    # Critical for training stability on deeper models
    attn_implementation="flash_attention_2",
)

# ==========================================
# PART 2: GUIDE Initialization
# ==========================================


def initialize_with_guide():
    logger.info(f"Loading Teacher ({TEACHER_ID}) for GUIDE initialization...")
    teacher = AutoModel.from_pretrained(
        TEACHER_ID, trust_remote_code=True, torch_dtype=torch.float32
    )

    logger.info("Initializing Blank Student...")
    student = ModernBertModel(student_config).to(dtype=torch.float32)

    # --- Step A: Compress Embeddings (PCA) ---
    logger.info("GUIDE Step A: Compressing Embeddings (768 -> 384)...")

    E_T = teacher.embeddings.tok_embeddings.weight.detach().numpy()

    pca = PCA(n_components=student_config.hidden_size)
    E_S_numpy = pca.fit_transform(E_T)

    # Projection Matrix M: [768, 384]
    M = pca.components_.T

    student.embeddings.tok_embeddings.weight.data = torch.tensor(E_S_numpy)
    logger.info("✅ Embeddings initialized via PCA.")

    # --- Step B: Project Layer 0 ---
    logger.info("GUIDE Step B: Projecting Layer 0 Weights...")

    t_layer0 = teacher.layers[0]
    s_layer0 = student.layers[0]

    # ---------------------------------------------------------
    # 1. Attention Inputs (Wqkv)
    # Teacher: [2304, 768] -> Student: [1152, 384]
    # ---------------------------------------------------------
    t_wqkv = t_layer0.attn.Wqkv.weight.detach().numpy()

    # Project Input (768->384)
    t_wqkv_proj = np.dot(t_wqkv, M)

    # Slice Output: Must handle Q, K, V blocks correctly
    # Teacher (12 heads): Q(768) | K(768) | V(768)
    # Student (6 heads):  Q(384) | K(384) | V(384)

    t_q, t_k, t_v = np.split(t_wqkv_proj, 3, axis=0)  # Each is [768, 384]

    # Slice each to keep only 6 heads (6 * 64 = 384 rows)
    s_q = t_q[:384, :]
    s_k = t_k[:384, :]
    s_v = t_v[:384, :]

    # Stack back together
    s_wqkv_final = np.concatenate([s_q, s_k, s_v], axis=0)  # [1152, 384]
    s_layer0.attn.Wqkv.weight.data = torch.tensor(s_wqkv_final)

    # ---------------------------------------------------------
    # 2. Attention Output (Wo)
    # Teacher: [768, 768] -> Student: [384, 384]
    # ---------------------------------------------------------
    t_wo = t_layer0.attn.Wo.weight.detach().numpy()

    # Slice Input columns (must match the 384 head outputs we kept above)
    t_wo_sliced = t_wo[:, :384]

    # Project Output rows (768 -> 384)
    s_wo_final = np.dot(t_wo_sliced.T, M).T
    s_layer0.attn.Wo.weight.data = torch.tensor(s_wo_final)

    # ---------------------------------------------------------
    # 3. MLP Input (Wi) - GeGLU Handling
    # Teacher: [2304, 768] -> Student: [1536, 384]
    # ---------------------------------------------------------
    t_wi = t_layer0.mlp.Wi.weight.detach().numpy()

    # Project Input (768 -> 384)
    t_wi_proj = np.dot(t_wi, M)

    # Slice Output: Must handle Gate/Value blocks correctly
    # Teacher Intermediate: 1152. Wi Output: 2304 (1152 Gate + 1152 Value)
    # Student Intermediate: 768.  Wi Output: 1536 (768 Gate + 768 Value)

    t_wi_gate, t_wi_val = np.split(t_wi_proj, 2, axis=0)  # Each is [1152, 384]

    # Keep first 768 neurons of Gate and Value
    s_wi_gate = t_wi_gate[:768, :]
    s_wi_val = t_wi_val[:768, :]

    s_wi_final = np.concatenate([s_wi_gate, s_wi_val], axis=0)  # [1536, 384]
    s_layer0.mlp.Wi.weight.data = torch.tensor(s_wi_final)

    # ---------------------------------------------------------
    # 4. MLP Output (Wo)
    # Teacher: [768, 1152] -> Student: [384, 768]
    # ---------------------------------------------------------
    t_mlp_wo = t_layer0.mlp.Wo.weight.detach().numpy()

    # Slice Input columns: Must match the 768 neurons we kept above
    # Teacher Input: 1152. Student Input: 768.
    t_mlp_wo_sliced = t_mlp_wo[:, :768]  # [768, 768]

    # Project Output rows (768 -> 384)
    s_mlp_wo_final = np.dot(t_mlp_wo_sliced.T, M).T  # [384, 768]

    s_layer0.mlp.Wo.weight.data = torch.tensor(s_mlp_wo_final)

    logger.info("✅ Layer 0 initialized via Projection + Slicing.")

    return student


# ==========================================
# PART 3: Save and Wrap
# ==========================================

if __name__ == "__main__":
    if os.path.exists(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH)

    # Initialize
    hf_model = initialize_with_guide()

    # Save Model
    logger.info(f"Saving initialized model to {OUTPUT_PATH}...")
    hf_model.save_pretrained(OUTPUT_PATH)

    # Save Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_ID)
    tokenizer.save_pretrained(OUTPUT_PATH)

    # Verify loading
    logger.info("Verifying load as SentenceTransformer...")

    # Force float32 loading to avoid flash attention mismatch during verification
    model = SentenceTransformer(
        OUTPUT_PATH, model_kwargs={"torch_dtype": torch.float32}
    )
    logger.info("Success! Model is ready for pre-training.")
