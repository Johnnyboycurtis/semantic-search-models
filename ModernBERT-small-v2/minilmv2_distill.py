"""
Stage 3: MiniLMv2 Multi-Head Self-Attention Relation Distillation.

Transfers Q-Q, K-K, and V-V self-attention relations from an upper-middle layer
of Alibaba-NLP/gte-modernbert-base (22 layers, 768-dim, 12 heads) into the final
layer of ModernBERT-small-v2 (12 layers, 384-dim, 6 heads) using 48 relation heads.
All student weights remain unfrozen.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

# --- Configuration ---
TEACHER_ID = "Alibaba-NLP/gte-modernbert-base"
STUDENT_PATH = "/content/drive/MyDrive/data/models/mlm-model-1b/checkpoint-10000"
TRAIN_FILE = "/content/drive/MyDrive/data/mlm_dataset.parquet"
OUTPUT_DIR = "/content/drive/MyDrive/data/models/minilmv2-student"

TEACHER_LAYER_IDX = 17  # Upper-middle layer (~80% depth of 22-layer teacher, 0-indexed)
RELATION_HEADS = 48     # Decouples 12-head teacher from 6-head student
MAX_LENGTH = 512
BATCH_SIZE = 32
GRAD_ACCUM_STEPS = 4    # Effective batch size = 128
LEARNING_RATE = 3e-4
NUM_STEPS = 20000
WARMUP_STEPS = 1000
LOG_EVERY = 100
SAVE_EVERY = 2500


class QKVHook:
    """Captures the fused Wqkv output tensor from a ModernBERT attention layer."""

    def __init__(self, module: nn.Module):
        self.qkv = None
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        self.qkv = output

    def remove(self):
        self.handle.remove()


def compute_self_attention_relations(
    qkv: torch.Tensor,
    num_relation_heads: int,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split fused Wqkv output into Q, K, V, reshape into relation heads,
    and return scaled dot-product relation probabilities for Q-Q, K-K, and V-V.
    """
    # Handle both padded (batch, seq_len, 3 * hidden) and unpadded (tokens, 3 * hidden)
    if qkv.dim() == 2:
        batch_size = attention_mask.shape[0]
        seq_len = attention_mask.shape[1]
        raise ValueError(
            "Received unpadded 2D QKV tensor; pass sequences without padding "
            "or pack chunks to MAX_LENGTH so attention_mask is all ones."
        )

    # Shape: batch size, sequence length, three times hidden dimension
    batch_size, seq_len, three_hidden = qkv.shape
    hidden_dim = three_hidden // 3
    rel_head_dim = hidden_dim // num_relation_heads

    # Split into Q, K, V; each has shape: batch size, sequence length, hidden dimension
    q, k, v = qkv.split(hidden_dim, dim=-1)

    def to_relation_probs(x: torch.Tensor) -> torch.Tensor:
        # Shape: batch size, relation heads, sequence length, relation head dimension
        x = x.view(batch_size, seq_len, num_relation_heads, rel_head_dim).transpose(1, 2)
        # Scaled dot-product relation scores: batch, relation heads, seq_len, seq_len
        scores = torch.matmul(x, x.transpose(-1, -2)) / math.sqrt(rel_head_dim)

        # Mask out any padding positions so pad tokens never receive attention mass
        # Shape: batch size, 1, 1, sequence length
        key_mask = attention_mask.unsqueeze(1).unsqueeze(2).bool()
        scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
        return F.softmax(scores, dim=-1)

    return to_relation_probs(q), to_relation_probs(k), to_relation_probs(v)


def minilmv2_relation_loss(
    t_rels: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    s_rels: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute average KL-divergence across Q-Q, K-K, and V-V relations over valid tokens."""
    # Shape: batch size, 1, sequence length, 1
    valid_query_mask = attention_mask.unsqueeze(1).unsqueeze(-1).float()
    num_valid_tokens = valid_query_mask.sum().clamp(min=1.0)
    num_rel_heads = t_rels[0].shape[1]

    total_kl = 0.0
    for t_prob, s_prob in zip(t_rels, s_rels):
        # KL(teacher || student) per token position
        kl = t_prob * (torch.log(t_prob.clamp(min=1e-9)) - torch.log(s_prob.clamp(min=1e-9)))
        # Sum over key dimension, mask out padded query positions, average over heads and tokens
        kl = (kl.sum(dim=-1, keepdim=True) * valid_query_mask).sum()
        total_kl = total_kl + kl / (num_rel_heads * num_valid_tokens)

    return total_kl / 3.0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading teacher: {TEACHER_ID}")
    teacher = AutoModel.from_pretrained(
        TEACHER_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    print(f"Loading student: {STUDENT_PATH}")
    student = AutoModel.from_pretrained(
        STUDENT_PATH, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student.train()
    for param in student.parameters():
        param.requires_grad = True  # Keep all student weights unfrozen

    tokenizer = AutoTokenizer.from_pretrained(TEACHER_ID)

    # Pack text into exact MAX_LENGTH chunks so there is zero padding waste
    # and ModernBERT keeps 3D (batch, seq_len, hidden) tensors in Wqkv hooks.
    raw_ds = load_dataset("parquet", data_files={"train": TRAIN_FILE})["train"]

    def pack_tokenize_fn(batch):
        tokenized_batch = tokenizer(batch["text"], truncation=False, return_token_type_ids=False)
        all_ids = [tok for ids in tokenized_batch["input_ids"] for tok in ids]
        total_len = (len(all_ids) // MAX_LENGTH) * MAX_LENGTH
        chunks = [all_ids[i : i + MAX_LENGTH] for i in range(0, total_len, MAX_LENGTH)]
        return {
            "input_ids": chunks,
            "attention_mask": [[1] * MAX_LENGTH for _ in chunks],
        }

    tokenized = raw_ds.map(pack_tokenize_fn, batched=True, remove_columns=raw_ds.column_names, num_proc=8)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    loader = DataLoader(tokenized, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    teacher_hook = QKVHook(teacher.layers[TEACHER_LAYER_IDX].attn.Wqkv)
    student_hook = QKVHook(student.layers[-1].attn.Wqkv)

    optimizer = torch.optim.AdamW(
        student.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), weight_decay=0.01
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=NUM_STEPS
    )

    step = 0
    optimizer.zero_grad()

    print("Starting MiniLMv2 relation distillation...")
    while step < NUM_STEPS:
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.no_grad():
                teacher(input_ids=input_ids, attention_mask=attention_mask)
                t_rels = compute_self_attention_relations(
                    teacher_hook.qkv, RELATION_HEADS, attention_mask
                )

            student(input_ids=input_ids, attention_mask=attention_mask)
            s_rels = compute_self_attention_relations(
                student_hook.qkv, RELATION_HEADS, attention_mask
            )

            loss = minilmv2_relation_loss(t_rels, s_rels, attention_mask) / GRAD_ACCUM_STEPS
            loss.backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            step += 1
            if step % LOG_EVERY == 0:
                print(f"Step {step}/{NUM_STEPS} | Relation KL Loss: {loss.item() * GRAD_ACCUM_STEPS:.4f}")

            if step % SAVE_EVERY == 0:
                ckpt_dir = f"{OUTPUT_DIR}/checkpoint-{step}"
                student.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                print(f"Saved checkpoint to {ckpt_dir}")

            if step >= NUM_STEPS:
                break

    teacher_hook.remove()
    student_hook.remove()
    student.save_pretrained(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
    print(f"Finished! Final model saved to {OUTPUT_DIR}/final")


if __name__ == "__main__":
    main()
