import os
import shutil

import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModel,
    AutoTokenizer,
    ModernBertConfig,
    ModernBertForMaskedLM,
    ModernBertModel,
)

# ==========================================
# PART 1: Configuration
# ==========================================

TEACHER_ID = "Alibaba-NLP/gte-modernbert-base"
OUTPUT_PATH = "./models/base-model"

# Map each of the 12 student layers to a corresponding layer in the 22-layer teacher.
# Both ModernBERT-base and our student use global_attn_every_n_layers=3, so teacher
# layers 0, 6, 12 use global attention while intermediate layers use local sliding-window
# attention. This mapping aligns global/local roles across depths:
#   Student global layers (0, 3, 6, 9) <- Teacher layers (0, 6, 12, 19)
#   Student local layers               <- Teacher local layers
LAYER_MAP = [0, 1, 2, 6, 7, 8, 12, 13, 14, 19, 20, 21]


def build_student_config(vocab_size: int, pad_token_id: int) -> ModernBertConfig:
    """Build the custom 'Deep & Narrow' 12-layer, 384-dim ModernBERT student config."""
    return ModernBertConfig(
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
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        # Note: attn_implementation="flash_attention_2" is intentionally omitted here;
        # it is a model-loading argument passed at training time with bfloat16.
    )


def _copy_weight(param: torch.nn.Parameter, array: np.ndarray) -> None:
    """Copy a NumPy weight array into a PyTorch Parameter in-place with matching dtype."""
    with torch.no_grad():
        param.copy_(torch.from_numpy(array).to(device=param.device, dtype=param.dtype))


# ==========================================
# PART 2: GUIDE Initialization
# ==========================================


def project_layer(
    t_layer: torch.nn.Module,
    s_layer: torch.nn.Module,
    M: np.ndarray,
    t_num_heads: int,
    s_num_heads: int,
) -> None:
    """Project a single teacher ModernBERT layer into a student layer using basis M."""
    t_hidden, s_hidden = M.shape
    head_dim = s_hidden // s_num_heads
    assert (
        t_hidden // t_num_heads == head_dim
    ), f"Head dim mismatch: teacher {t_hidden // t_num_heads} vs student {head_dim}"

    # ---------------------------------------------------------
    # 1. Attention Inputs (Wqkv) & Output (Wo)
    # Teacher: Wqkv [2304, 768], Wo [768, 768]
    # Student: Wqkv [1152, 384], Wo [384, 384]
    # ---------------------------------------------------------
    t_wqkv = t_layer.attn.Wqkv.weight.detach().numpy()
    t_wo = t_layer.attn.Wo.weight.detach().numpy()

    assert t_wqkv.shape == (
        3 * t_hidden,
        t_hidden,
    ), f"Unexpected Wqkv shape: {t_wqkv.shape}"
    assert t_wo.shape == (t_hidden, t_hidden), f"Unexpected Wo shape: {t_wo.shape}"

    # Project input space of Wqkv (768 -> 384) and output space of Wo (768 -> 384)
    t_wqkv_proj = np.dot(t_wqkv, M)  # [3 * t_hidden, s_hidden]
    t_wo_proj = np.dot(M.T, t_wo)  # [s_hidden, t_hidden]

    # Split Q, K, V blocks and reshape into per-head slices: [t_num_heads, head_dim, s_hidden]
    t_q, t_k, t_v = np.split(t_wqkv_proj, 3, axis=0)
    assert t_q.shape == (t_hidden, s_hidden)

    t_q_heads = t_q.reshape(t_num_heads, head_dim, s_hidden)
    t_k_heads = t_k.reshape(t_num_heads, head_dim, s_hidden)
    t_v_heads = t_v.reshape(t_num_heads, head_dim, s_hidden)
    t_wo_heads = t_wo_proj.reshape(s_hidden, t_num_heads, head_dim)

    # Rank teacher heads by joint L2 norm (Q/K/V + Wo) and keep top-k in original order
    head_scores = np.sqrt(
        np.sum(t_q_heads**2, axis=(1, 2))
        + np.sum(t_k_heads**2, axis=(1, 2))
        + np.sum(t_v_heads**2, axis=(1, 2))
        + np.sum(t_wo_heads**2, axis=(0, 2))
    )
    top_heads = np.sort(np.argsort(head_scores)[-s_num_heads:])

    s_q = t_q_heads[top_heads].reshape(s_num_heads * head_dim, s_hidden)
    s_k = t_k_heads[top_heads].reshape(s_num_heads * head_dim, s_hidden)
    s_v = t_v_heads[top_heads].reshape(s_num_heads * head_dim, s_hidden)

    s_wqkv_final = np.concatenate([s_q, s_k, s_v], axis=0)  # [1152, 384]
    s_wo_final = t_wo_heads[:, top_heads, :].reshape(
        s_hidden, s_num_heads * head_dim
    )  # [384, 384]

    assert s_wqkv_final.shape == tuple(s_layer.attn.Wqkv.weight.shape)
    assert s_wo_final.shape == tuple(s_layer.attn.Wo.weight.shape)

    _copy_weight(s_layer.attn.Wqkv.weight, s_wqkv_final)
    _copy_weight(s_layer.attn.Wo.weight, s_wo_final)

    # ---------------------------------------------------------
    # 2. MLP Input (Wi) & Output (Wo) - GeGLU Handling
    # Teacher: Wi [2304, 768] (1152 Gate + 1152 Value), Wo [768, 1152]
    # Student: Wi [1536, 384] (768 Gate + 768 Value),   Wo [384, 768]
    # ---------------------------------------------------------
    t_wi = t_layer.mlp.Wi.weight.detach().numpy()
    t_mlp_wo = t_layer.mlp.Wo.weight.detach().numpy()

    t_inter = t_mlp_wo.shape[1]
    s_inter = s_layer.mlp.Wo.weight.shape[1]

    assert t_wi.shape == (2 * t_inter, t_hidden), f"Unexpected Wi shape: {t_wi.shape}"
    assert t_mlp_wo.shape == (
        t_hidden,
        t_inter,
    ), f"Unexpected MLP Wo shape: {t_mlp_wo.shape}"

    # Project input space of Wi (768 -> 384) and output space of MLP Wo (768 -> 384)
    t_wi_proj = np.dot(t_wi, M)  # [2 * t_inter, s_hidden]
    t_mlp_wo_proj = np.dot(M.T, t_mlp_wo)  # [s_hidden, t_inter]

    # Split Gate/Value blocks to keep paired GeGLU neurons aligned
    t_wi_gate, t_wi_val = np.split(t_wi_proj, 2, axis=0)
    assert t_wi_gate.shape == (t_inter, s_hidden)
    assert t_wi_val.shape == (t_inter, s_hidden)

    # Rank GeGLU neurons by joint L2 norm (Wi gate/value + Wo column) and keep top-k in order
    neuron_scores = np.sqrt(
        np.sum(t_wi_gate**2, axis=1)
        + np.sum(t_wi_val**2, axis=1)
        + np.sum(t_mlp_wo_proj**2, axis=0)
    )
    top_neurons = np.sort(np.argsort(neuron_scores)[-s_inter:])

    s_wi_gate = t_wi_gate[top_neurons, :]
    s_wi_val = t_wi_val[top_neurons, :]
    s_wi_final = np.concatenate([s_wi_gate, s_wi_val], axis=0)  # [1536, 384]
    s_mlp_wo_final = t_mlp_wo_proj[:, top_neurons]  # [384, 768]

    assert s_wi_final.shape == tuple(s_layer.mlp.Wi.weight.shape)
    assert s_mlp_wo_final.shape == tuple(s_layer.mlp.Wo.weight.shape)

    _copy_weight(s_layer.mlp.Wi.weight, s_wi_final)
    _copy_weight(s_layer.mlp.Wo.weight, s_mlp_wo_final)


def initialize_with_guide() -> ModernBertModel:
    print(f"Loading Teacher ({TEACHER_ID}) for GUIDE initialization...")
    teacher = AutoModel.from_pretrained(
        TEACHER_ID, trust_remote_code=True, torch_dtype=torch.float32
    )

    student_config = build_student_config(
        vocab_size=teacher.config.vocab_size,
        pad_token_id=teacher.config.pad_token_id,
    )
    assert len(LAYER_MAP) == student_config.num_hidden_layers

    print("Initializing Blank Student...")
    student = ModernBertModel(student_config).to(dtype=torch.float32)

    # --- Step A: Compress Embeddings via Frequency-Weighted Exact SVD ---
    print("GUIDE Step A: Compressing Embeddings (768 -> 384) via Frequency-Weighted SVD...")

    # 1. Count token frequencies on a fast 50k-row sample of the corpus
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_ID)
    sample_ds = load_dataset(
        "parquet",
        data_files={"train": "data/combined_mlm_dataset.parquet"},
        split="train[:50000]",
    )

    counts = np.zeros(teacher.config.vocab_size, dtype=np.float64)
    for i in range(0, len(sample_ds), 1000):
        batch_ids = tokenizer(
            sample_ds[i : i + 1000]["text"], truncation=True, max_length=1024
        )["input_ids"]
        flat_ids = np.fromiter((tid for seq in batch_ids for tid in seq), dtype=np.int64)
        counts += np.bincount(flat_ids, minlength=teacher.config.vocab_size)

    # Smooth with sqrt(count + 1); zero out [PAD] so padding doesn't steer the basis
    weights = np.sqrt(counts + 1.0)
    weights[teacher.config.pad_token_id] = 0.0

    # 2. Exact SVD on frequency-weighted embeddings (deterministic)
    E_T = teacher.embeddings.tok_embeddings.weight.detach().cpu().numpy()
    E_weighted = E_T * weights[:, None]

    _, S, Vt = np.linalg.svd(E_weighted, full_matrices=False)
    M = Vt[: student_config.hidden_size].T  # Orthonormal [768, 384] basis

    retained = (S[: student_config.hidden_size] ** 2).sum() / (S**2).sum()
    print(f"Frequency-Weighted SVD retained variance: {retained:.2%}")

    # 3. Project the UNWEIGHTED embeddings into basis M so token norms stay intact
    E_S = E_T @ M
    _copy_weight(student.embeddings.tok_embeddings.weight, E_S)

    # Safeguard: ensure [PAD] token embedding remains strictly zero.
    with torch.no_grad():
        student.embeddings.tok_embeddings.weight[teacher.config.pad_token_id].zero_()
    print("Embeddings initialized via frequency-weighted SVD projection.")

    # --- Step B: Project All 12 Student Layers ---
    print(
        f"GUIDE Step B: Projecting all {student_config.num_hidden_layers} layers "
        f"using teacher mapping {LAYER_MAP}..."
    )
    for s_idx, t_idx in enumerate(LAYER_MAP):
        project_layer(
            teacher.layers[t_idx],
            student.layers[s_idx],
            M,
            t_num_heads=teacher.config.num_attention_heads,
            s_num_heads=student_config.num_attention_heads,
        )

    print("All 12 layers initialized via Projection + Norm-Ranked Slicing.")

    return student


# ==========================================
# PART 3: Save and Verify
# ==========================================


def main() -> None:
    if os.path.exists(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH)

    # Initialize
    hf_model = initialize_with_guide()

    # Save Model
    print(f"Saving initialized model to {OUTPUT_PATH}...")
    hf_model.save_pretrained(OUTPUT_PATH)

    # Save Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_ID)
    tokenizer.save_pretrained(OUTPUT_PATH)

    # Verify loading for Phase 1 (MLM Pre-training)
    print("Verifying load as ModernBertForMaskedLM (tie_word_embeddings=False)...")
    mlm_model = ModernBertForMaskedLM.from_pretrained(
        OUTPUT_PATH, tie_word_embeddings=False, torch_dtype=torch.float32
    )
    mlm_model.eval()
    sample_inputs = tokenizer(
        "The quick brown fox jumps over the lazy dog.", return_tensors="pt"
    )
    with torch.no_grad():
        outputs = mlm_model(**sample_inputs, labels=sample_inputs["input_ids"])

    seq_len = sample_inputs["input_ids"].shape[1]
    expected_shape = (1, seq_len, mlm_model.config.vocab_size)
    assert (
        outputs.logits.shape == expected_shape
    ), f"Expected logits shape {expected_shape}, got {outputs.logits.shape}"
    assert torch.isfinite(
        outputs.loss
    ), "MLM dummy forward pass produced non-finite loss."
    assert torch.isfinite(
        outputs.logits
    ).all(), "MLM dummy forward pass produced non-finite logits."
    print(
        f"MLM verification passed (logits={tuple(outputs.logits.shape)}, loss={outputs.loss.item():.4f})."
    )

    # Verify loading for Phase 2/3 (Contrastive Fine-tuning)
    print("Verifying load as SentenceTransformer...")
    # Force float32 loading during CPU/offline verification
    st_model = SentenceTransformer(
        OUTPUT_PATH, model_kwargs={"torch_dtype": torch.float32}
    )
    emb = st_model.encode("The quick brown fox jumps over the lazy dog.")
    assert emb.shape == (
        st_model[0].auto_model.config.hidden_size,
    ), f"Unexpected SentenceTransformer embedding shape: {emb.shape}"
    print("Success! Model is ready for pre-training.")


if __name__ == "__main__":
    main()
