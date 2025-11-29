# ==============================================================================
#           Benchmarking Models on BeIR/hotpotqa Dataset
# ==============================================================================
#
# PURPOSE:
# This script evaluates the performance of multiple sentence embedding models on
# the HotpotQA information retrieval task from the BeIR benchmark.
#
# WHAT IT DOES:
# 1.  Defines a dictionary of models to be benchmarked.
# 2.  Loads and prepares the `BeIR/hotpotqa` dataset.
# 3.  Initializes the `InformationRetrievalEvaluator`.
# 4.  Loops through each model, runs the evaluation, and stores the results.
# 5.  Prints a clean, formatted summary table and saves results to a CSV file.
#
# ==============================================================================

import logging
import torch
import random
import csv
from datetime import datetime
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator

# --- Configuration ---
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

# --- Step 1: Define Models to Benchmark ---
# This dictionary contains the models we want to compare.
# Please double-check that these paths are correct.
models_to_benchmark = {
    "distilled-mnrl-ModernBERT-small": "ModernBERT-small/distilled-mnrl-ModernBERT-small/final", # checkpoint-6798
    "distilled-mnrl-ModernBERT-small-checkpoint-6798": "ModernBERT-small/distilled-kldiv-ModernBERT-small/checkpoint-7032", # 
    "trained-sts-fine-tuned": "./ModernBERT-small/sts-tuned-modernbert-small/final",
    "distilled-trained-sts-fine-tuned": "./ModernBERT-small/distilled-sts-tuned-modernbert-small/final",
    "distilled-ModernBERT-small": "ModernBERT-small/distilled-kldiv-ModernBERT-small/checkpoint-2266",
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
}

# Define the device to use
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

# --- Step 2: Load and Prepare the HotpotQA Dataset ---
dataset_name = "BeIR/hotpotqa"
logging.info(f"Loading dataset: {dataset_name}")

# Load the corpus, queries, and relevance judgments (qrels)
corpus_data = load_dataset(dataset_name, "corpus", split="corpus")
queries_data = load_dataset(dataset_name, "queries", split="queries")
relevant_docs_data = load_dataset("BeIR/hotpotqa-qrels", split="test")

# For this dataset, we want to concatenate the title and texts for the corpus
logging.info("Preprocessing corpus by concatenating title and text.")
corpus_data = corpus_data.map(lambda x: {'text': x['title'] + " " + x['text']}, remove_columns=['title'])


# --- [NEW] Shrink the Corpus for Faster Evaluation ---
# The full HotpotQA corpus is massive. For rapid, iterative benchmarking,
# we create a smaller subset that is guaranteed to contain all the relevant
# documents from the test set, plus a random sample of distractor documents.
logging.info("Shrinking corpus for faster evaluation...")
required_corpus_ids = set(relevant_docs_data["corpus-id"])
if len(corpus_data) > 30000:
    all_corpus_ids = corpus_data["_id"]
    # Create a pool of IDs that are not in the required set
    sample_pool = [cid for cid in all_corpus_ids if cid not in required_corpus_ids]
    # Add 30,000 random documents as distractors
    k = min(30000, len(sample_pool))
    required_corpus_ids.update(random.sample(sample_pool, k=k))

corpus_data = corpus_data.filter(lambda x: x["_id"] in required_corpus_ids)
logging.info(f"Final shrunk corpus size: {len(corpus_data):,}")



# Convert the datasets to dictionaries for the evaluator
corpus = dict(zip(corpus_data["_id"], corpus_data["text"]))
queries = dict(zip(queries_data["_id"], queries_data["text"]))
relevant_docs = {}
for qrel in relevant_docs_data:
    qid = str(qrel["query-id"])
    corpus_id = str(qrel["corpus-id"])
    if qid not in relevant_docs:
        relevant_docs[qid] = set()
    relevant_docs[qid].add(corpus_id)

logging.info(f"Dataset loaded: {len(corpus):,} documents and {len(queries):,} queries.")


# --- Step 3: Initialize the Evaluator ---
logging.info("Initializing the InformationRetrievalEvaluator...")
evaluator_name = "BeIR-hotpotqa-test"
ir_evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    name=evaluator_name,
    show_progress_bar=True,
)

# --- Step 4: Run the Benchmark Loop ---
print("\n" + "="*80)
print(" 🚀 STARTING HOTPOTQA INFORMATION RETRIEVAL BENCHMARK 🚀")
print("="*80 + "\n")

all_results = []

for model_name, model_path in models_to_benchmark.items():
    print(f"\n--- Evaluating Model: '{model_name}' ---")
    
    try:
        # Load the SentenceTransformer model
        model = SentenceTransformer(model_path, device=device)
        
        # Run the evaluation
        results = ir_evaluator(model)
        
        # Store results for the final summary table
        results_summary = {
            "Model": model_name,
            "nDCG@10": results[f"{evaluator_name}_cosine_ndcg@10"],
            "MAP@100": results[f"{evaluator_name}_cosine_map@100"],
            "Recall@10": results[f"{evaluator_name}_cosine_recall@10"],
            "Precision@10": results[f"{evaluator_name}_cosine_precision@10"],
        }
        all_results.append(results_summary)
        
        print(f"Evaluation for '{model_name}' complete.")

    except Exception as e:
        logging.error(f"Failed to evaluate model {model_name}. Error: {e}")
        # Add a placeholder result so the script can continue
        all_results.append({
            "Model": model_name,
            "nDCG@10": "ERROR",
            "MAP@100": "ERROR",
            "Recall@10": "ERROR",
            "Precision@10": "ERROR",
        })

    print("-" * 40)


# --- Step 5: Display and Save Final Results ---
print("\n" + "="*80)
print(" ✅ BENCHMARK COMPLETE ✅")
print("="*80 + "\n")

# Save results to a CSV file for tracking
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
csv_filename = f"hotpotqa_benchmark_results_{timestamp}.csv"
if all_results:
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = all_results[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    logging.info(f"Benchmark results saved to {csv_filename}")

# Print a formatted summary table
if all_results:
    header = all_results[0].keys()
    print(f"{'Model':<45} | {'nDCG@10':<10} | {'MAP@100':<10} | {'Recall@10':<10} | {'Precision@10':<12}")
    print(f"{'-'*45} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*12}")

    for res in all_results:
        ndcg_val = f"{res['nDCG@10']:.4f}" if isinstance(res['nDCG@10'], float) else res['nDCG@10']
        map_val = f"{res['MAP@100']:.4f}" if isinstance(res['MAP@100'], float) else res['MAP@100']
        recall_val = f"{res['Recall@10']:.4f}" if isinstance(res['Recall@10'], float) else res['Recall@10']
        precision_val = f"{res['Precision@10']:.4f}" if isinstance(res['Precision@10'], float) else res['Precision@10']
        print(f"{res['Model']:<45} | {ndcg_val:<10} | {map_val:<10} | {recall_val:<10} | {precision_val:<12}")
else:
    print("No models were evaluated.")

print("\n" + "="*80)
