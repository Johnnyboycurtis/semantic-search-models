"""
Pre-tokenize and filter FineWiki (English) for Optional Phase 1B (4,096-token MLM).

Saves the filtered Arrow dataset (>= 1,500 tokens per doc) to Google Drive so
pre_train_phase1b_optional.py can load it in seconds via load_from_disk()
without repeating the ~2-hour tokenization step.
"""

import os

fine_wiki_exists = "finewiki_en.parquet" in os.listdir("drive/MyDrive/data/")
print(f"Fine-wiki exists: {fine_wiki_exists}")

from datasets import load_dataset
from transformers import AutoTokenizer

TRAIN_FILE = "drive/MyDrive/data/finewiki_en.parquet"
MODEL_PATH = "drive/MyDrive/data/models/mlm-model/final"
MAX_LENGTH = 4096
MIN_DOC_TOKENS = 1500

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


print(tokenizer.encode("hello"))
print(tokenizer.decode([50281, 25521, 50282]))


dataset = load_dataset("parquet", data_files={"train": TRAIN_FILE}, num_proc=8)["train"]
print(f"Before filtering: {len(dataset):,} rows")


tokenized = dataset.map(
    lambda ex: tokenizer(ex["text"], truncation=True, max_length=MAX_LENGTH),
    batched=True,
    remove_columns=dataset.column_names,
    num_proc=6,
)


filtered = tokenized.filter(lambda x: len(x["input_ids"]) >= MIN_DOC_TOKENS, num_proc=8)
print(f"After filtering (>= {MIN_DOC_TOKENS} tokens): {len(filtered):,} rows")

filtered.save_to_disk("/content/drive/MyDrive/data/finewiki_4096_filtered")


# from datasets import load_from_disk

# tokenized_datasets = load_from_disk("/content/drive/MyDrive/data/finewiki_4096_filtered")