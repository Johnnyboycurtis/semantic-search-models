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
    phil_ds = load_dataset("johnnyboycurtis/Philosophical-Triplets-Retrieval")["train"]

    def flatten_phil(batch):
        # Combines the entry titles and the contents into a flat list
        return {
            "text": batch["query"]
            + batch["positive"]
            + batch["negative"]
            + batch["text_chunk"]
        }

    print("Mapping Philosophy Dataset...")
    phil_flat = phil_ds.map(
        flatten_phil, batched=True, remove_columns=phil_ds.column_names
    )
    phil_flat.to_parquet("data/johnnyboycurtis_philosophy.parquet")




    # 1. Load the dataset
    print("\n🚀 Loading NPR...")
    npr_ds = load_dataset("sentence-transformers/npr")["train"]

    # 2. Define the mapping function
    def npr_concatenate_title_and_body(example):
        """
        Concatenates the 'title' and 'body' fields into a new 'text' field
        following the specified format.
        """
        title = example['title']
        body = example['body']
        
        # Create the desired output format
        output_str = f"# {title} \n\n {body}"
        
        # Return the example dictionary with the new field
        return {"text": output_str}

    # 3. Apply the map function
    # We use batched=False (the default) because we are processing one example at a time.
    # We also use remove_columns to clean up the dataset and keep only the new 'text' column.
    print("Mapping NPR Dataset...")
    npr_flat = npr_ds.map(
        npr_concatenate_title_and_body,
        remove_columns=npr_ds.column_names,
        #batched=True
    )
    npr_flat.to_parquet("data/npr_paragraphs.parquet")

    print("\n✅ Success! Files created:")
    print(f"- data/msmarco_triplets.parquet ({len(ms_flat)} rows)")
    print(f"- data/johnnyboycurtis_philosophy.parquet ({len(phil_flat)} rows)")
    print(f"- data/npr.parquet ({len(npr_flat)} rows)")


# In your training script:
def load_and_combine_data():
    # Load both local parquets
    ms_ds = load_dataset(
        "parquet", data_files={"train": "data/msmarco_triplets.parquet"}
    )["train"]
    phil_ds = load_dataset(
        "parquet", data_files={"train": "data/johnnyboycurtis_philosophy.parquet"}
    )["train"]
    npr_ds = load_dataset(
        "parquet", data_files={"train": "data/npr_paragraphs.parquet"}
    )["train"]

    # Concatenate them for a unified training run
    from datasets import concatenate_datasets

    combined_dataset = concatenate_datasets([ms_ds, phil_ds, npr_ds])

    # Shuffle to ensure the model alternates between 'Search' style and 'Academic' style text
    output_file_name = "data/combined_mlm_dataset.parquet"
    combined_dataset = combined_dataset.shuffle(seed=123)
    combined_dataset.to_parquet(output_file_name)
    print(f"Saved combined dataset: {output_file_name}")


if __name__ == "__main__":
    process_datasets()

    load_and_combine_data()
