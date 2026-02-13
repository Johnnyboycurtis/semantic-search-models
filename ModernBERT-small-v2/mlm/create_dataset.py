import os
from datasets import load_dataset

# 1. SETUP
os.makedirs("data", exist_ok=True)


def process_datasets():
    # --- DATASET 1: MS MARCO Triplets ---
    print("🚀 Loading MS MARCO Triplets...")
    ms_ds = load_dataset(
        "sentence-transformers/msmarco-msmarco-MiniLM-L6-v3", "triplet"
    )["train"]

    def flatten_triplets(batch):
        # Concatenates all three columns into one long list of strings
        return {"text": batch["query"] + batch["positive"] + batch["negative"]}

    print("Mapping MS MARCO (Tripling row count for MLM)...")
    ms_flat = ms_ds.map(
        flatten_triplets, batched=True, remove_columns=ms_ds.column_names
    )
    ms_flat.to_parquet("data/msmarco_triplets.parquet")

    # --- DATASET 2: Stanford Encyclopedia of Philosophy ---
    print("\n🚀 Loading Stanford Philosophy...")
    phil_ds = load_dataset("johnnyboycurtis/stanford_encyclopedia_of_philosophy")[
        "train"
    ]

    def flatten_phil(batch):
        # Combines the entry titles and the contents into a flat list
        return {"text": batch["entry"] + batch["contents"]}

    print("Mapping Philosophy Dataset...")
    phil_flat = phil_ds.map(
        flatten_phil, batched=True, remove_columns=phil_ds.column_names
    )
    phil_flat.to_parquet("data/johnnyboycurtis_philosophy.parquet")

    print("\n✅ Success! Files created:")
    print(f"- data/msmarco_triplets.parquet ({len(ms_flat)} rows)")
    print(f"- data/johnnyboycurtis_philosophy.parquet ({len(phil_flat)} rows)")


# In your training script:
def load_and_combine_data():
    # Load both local parquets
    ms_ds = load_dataset(
        "parquet", data_files={"train": "data/msmarco_triplets.parquet"}
    )["train"]
    phil_ds = load_dataset(
        "parquet", data_files={"train": "data/johnnyboycurtis_philosophy.parquet"}
    )["train"]

    # Concatenate them for a unified training run
    from datasets import concatenate_datasets

    combined_dataset = concatenate_datasets([ms_ds, phil_ds])

    # Shuffle to ensure the model alternates between 'Search' style and 'Academic' style text
    output_file_name = "data/combined_mlm_dataset.parquet"
    combined_dataset = combined_dataset.shuffle(seed=42)
    combined_dataset.to_parquet(output_file_name)
    print(f"Saved combined dataset: {output_file_name}")


if __name__ == "__main__":
    process_datasets()

    load_and_combine_data()
