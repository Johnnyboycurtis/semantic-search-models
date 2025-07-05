import random
import logging
import torch
import csv
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from datasets import load_dataset

# --- Configuration ---
# Set up logging to show informational messages.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- [FIX] Explicitly define the device ---
# This is the key fix. We will explicitly tell SentenceTransformers to use the GPU
# if available. This avoids ambiguity and prevents tensors from being accidentally
# created on the CPU during the evaluation process.
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

# --- Step 1: Define Models to Benchmark ---
# We create a dictionary to hold the models we want to compare.
# The loop below will iterate through this dictionary.
# I've added 'bge-small-en-v1.5' as another strong competitor.
models_to_benchmark = {
    "My Small ModernBERT": "./output/training-small-modernbert/final-best",
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
}

# --- Step 2: Load and Prepare the Dataset ---
# You can easily swap this out for another BEIR dataset, like "BeIR/scifact"
dataset_name = "BeIR/webis-touche2020"
logging.info(f"Loading dataset: {dataset_name}")
logging.info("Loading dataset: BeIR/webis-touche2020")
corpus_data = load_dataset("BeIR/webis-touche2020", "corpus", split="corpus")
queries_data = load_dataset("BeIR/webis-touche2020", "queries", split="queries")
relevant_docs_data = load_dataset("BeIR/webis-touche2020-qrels", split="test")

# For this dataset, we want to concatenate the title and texts for the corpus
logging.info("Preprocessing corpus by concatenating title and text.")
corpus_data = corpus_data.map(lambda x: {'text': x['title'] + " " + x['text']}, remove_columns=['title'])

# --- Step 3: Shrink Corpus and Prepare for Evaluator ---
# This section correctly shrinks the corpus to include all relevant documents
# plus a random sample of 30,000 other documents for a faster, yet still
# meaningful, evaluation.

logging.info("Shrinking corpus to relevant docs + 30,000 random docs.")
# 1. Start with the set of all known relevant documents.
required_corpus_ids = set(map(str, relevant_docs_data["corpus-id"]))

# 2. ADD 30,000 random documents to that set.
# Note: We check if the corpus is large enough to sample from.
if len(corpus_data) > 30000:
    # Get a list of all corpus IDs to sample from
    all_corpus_ids = corpus_data["_id"]
    # Filter out IDs that are already required to avoid sampling duplicates
    sample_pool = [cid for cid in all_corpus_ids if cid not in required_corpus_ids]
    # Ensure we don't try to sample more than available
    k = min(30000, len(sample_pool))
    required_corpus_ids.update(random.sample(sample_pool, k=k))

corpus_data = corpus_data.filter(lambda x: x["_id"] in required_corpus_ids)
logging.info(f"Final shrunk corpus size: {len(corpus_data):,}")

# Convert the datasets to the dictionary format required by the evaluator
corpus = dict(zip(corpus_data["_id"], corpus_data["text"]))
queries = dict(zip(queries_data["_id"], queries_data["text"]))
relevant_docs = {}
for qrel in relevant_docs_data:
    qid = str(qrel["query-id"])
    corpus_id = str(qrel["corpus-id"])
    if qid not in relevant_docs:
        relevant_docs[qid] = set()
    relevant_docs[qid].add(corpus_id)

# --- Step 4: Initialize the Evaluator ---
# The evaluator is created once and then used for each model.
logging.info("Initializing the InformationRetrievalEvaluator...")
evaluator_name = f"{dataset_name.split('/')[-1]}-test"
ir_evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    name=evaluator_name,
    show_progress_bar=True,
    # Key metrics for IR.
    ndcg_at_k=[10, 100],
    map_at_k=[100],
)

# --- Step 5: Run the Benchmark Loop ---
print("\n" + "="*80)
print(" 🚀 STARTING INFORMATION RETRIEVAL BENCHMARK 🚀")
print("="*80 + "\n")

all_results = []

for model_name, model_path in models_to_benchmark.items():
    print(f"\n--- Evaluating Model: '{model_name}' ---")
    
    # Load the SentenceTransformer model, explicitly passing the device.
    model = SentenceTransformer(model_path, device=device)
    
    # Run the evaluation
    results = ir_evaluator(model)
    
    # Store results for final summary
    results_summary = {
        "Model": model_name,
        "nDCG@10": results[f"{evaluator_name}_cosine_ndcg@10"],
        "MAP@100": results[f"{evaluator_name}_cosine_map@100"],
        "Recall@10": results[f"{evaluator_name}_cosine_recall@10"],
    }
    all_results.append(results_summary)
    
    print(f"\nResults for '{model_name}' processed.")
    print("-" * 40)

# --- Step 6: Save and Display Final Results ---

# Save results to a CSV file for tracking
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
csv_filename = f"benchmark_results_{timestamp}.csv"
with open(csv_filename, 'w', newline='') as csvfile:
    fieldnames = all_results[0].keys()
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_results)
logging.info(f"Benchmark results saved to {csv_filename}")

# Display a formatted table in the console
print("\n" + "="*80)
print(" ✅ BENCHMARK COMPLETE ✅")
print("="*80 + "\n")

# Print header
header = all_results[0].keys()
print(f"{'Model':<30} | {'nDCG@10':<10} | {'MAP@100':<10} | {'Recall@10':<10}")
print(f"{'-'*30} | {'-'*10} | {'-'*10} | {'-'*10}")

# Print rows
for res in all_results:
    print(f"{res['Model']:<30} | {res['nDCG@10']:<10.4f} | {res['MAP@100']:<10.4f} | {res['Recall@10']:<10.4f}")

print("\n" + "="*80)
