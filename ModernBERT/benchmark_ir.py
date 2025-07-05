# ==============================================================================
#          Benchmarking Sentence Embedding Models for Information Retrieval
# ==============================================================================
#
# PURPOSE:
# This script (benchmark_ir.py) evaluates the performance of sentence embedding
# models on a standard Information Retrieval (IR) task. It compares our
# custom-trained "Small ModernBERT" against a strong, general-purpose model,
# `BAAI/bge-small-en-v1.5`.
#
# The goal is to simulate a real-world use case: given a user query, how
# well can each model find the most relevant documents from a large database?
#
# WHAT IT DOES:
# 1. Loads two models: our trained model and the BGE competitor.
# 2. Loads the SciFact dataset, a standard benchmark for IR.
# 3. Prepares the dataset into the format required by the evaluator:
#    - `queries`: A dictionary of {query_id: query_text}
#    - `corpus`: A dictionary of {document_id: document_text}
#    - `relevant_docs`: A dictionary of {query_id: set_of_relevant_document_ids}
# 4. Initializes the `InformationRetrievalEvaluator`.
# 5. Runs the evaluation for each model and prints the results.
#
# BEFORE YOU RUN:
# 1. Make sure your trained model exists at the path specified in `my_model_path`.
# 2. Install necessary libraries:
#    pip install sentence-transformers datasets
#
# ==============================================================================

import logging
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from datasets import load_dataset
import torch

if torch.cuda.is_available():
    print("GPU (CUDA) is available for PyTorch.")
    print(f"Number of available GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU device name: {torch.cuda.get_device_name(0)}") # Get name of first GPU
else:
    print("GPU (CUDA) is not available for PyTorch. PyTorch will run on CPU.")

# --- Configuration ---
# Set up logging to show informational messages.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the models we want to compare.
import random
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from datasets import load_dataset


# Load the Touche-2020 IR dataset (https://huggingface.co/datasets/BeIR/webis-touche2020, https://huggingface.co/datasets/BeIR/webis-touche2020-qrels)
corpus = load_dataset("BeIR/webis-touche2020", "corpus", split="corpus")
queries = load_dataset("BeIR/webis-touche2020", "queries", split="queries")
relevant_docs_data = load_dataset("BeIR/webis-touche2020-qrels", split="test")

# For this dataset, we want to concatenate the title and texts for the corpus
corpus = corpus.map(lambda x: {'text': x['title'] + " " + x['text']}, remove_columns=['title'])

# Shrink the corpus size heavily to only the relevant documents + 30,000 random documents
required_corpus_ids = set(map(str, relevant_docs_data["corpus-id"]))
required_corpus_ids |= set(random.sample(corpus["_id"], k=30_000))
corpus = corpus.filter(lambda x: x["_id"] in required_corpus_ids)

# Convert the datasets to dictionaries
corpus = dict(zip(corpus["_id"], corpus["text"]))  # Our corpus (cid => document)
queries = dict(zip(queries["_id"], queries["text"]))  # Our queries (qid => question)
relevant_docs = {}  # Query ID to relevant documents (qid => set([relevant_cids])
for qid, corpus_ids in zip(relevant_docs_data["query-id"], relevant_docs_data["corpus-id"]):
    qid = str(qid)
    corpus_ids = str(corpus_ids)
    if qid not in relevant_docs:
        relevant_docs[qid] = set()
    relevant_docs[qid].add(corpus_ids)

# Given queries, a corpus and a mapping with relevant documents, the InformationRetrievalEvaluator computes different IR metrics.
ir_evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    name="BeIR-touche2020-subset-test",
    show_progress_bar=True
)


# --- Step 3: Run the Benchmark Loop ---
# Now we'll iterate through our dictionary of models, run the evaluation for
# each one, and print the results.
print("\n" + "="*80)
print(" 🚀 STARTING INFORMATION RETRIEVAL BENCHMARK 🚀")
print("="*80 + "\n")


# Load the SentenceTransformer model from the path or Hugging Face Hub.
#model_path = "./output/training-small-modernbert/final-best"
model_path = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(model_path, model_kwargs={"torch_dtype": torch.bfloat16})
print(model)

# The evaluator object is callable. Passing the model to it will trigger
# the full evaluation process. This can take some time, especially for
# the first model, as it needs to encode the entire corpus.
results = ir_evaluator(model)
print(results)

# --- Step 6: Display Final Results ---
print("\n" + "="*80)
print(" ✅ EVALUATION COMPLETE ✅")
print("="*80 + "\n")

print(f"Results for '{model_path}':\n")
# [FIX] Dynamically construct the keys using the evaluator's name attribute.
# This makes the script robust to changes in the evaluator's name.
print(f"  - nDCG@10:      {results[f'{ir_evaluator.name}_cosine_ndcg@10']:.4f}")
print(f"  - MAP@100:      {results[f'{ir_evaluator.name}_cosine_map@100']:.4f}")
print(f"  - Recall@10:    {results[f'{ir_evaluator.name}_cosine_recall@10']:.4f}")
print(f"  - Precision@10: {results[f'{ir_evaluator.name}_cosine_precision@10']:.4f}")
print(f"  - Accuracy@10:  {results[f'{ir_evaluator.name}_cosine_accuracy@10']:.4f}")


print("\n" + "="*80)

# --- Understanding the Key Metrics ---
#
# nDCG@10 (Normalized Discounted Cumulative Gain at 10):
# This is often the most important metric for search ranking. It measures the
# quality of the top 10 results. It rewards models for placing highly relevant
# documents at the very top of the results list. A higher nDCG means better
# ranking quality.
#
# MAP@100 (Mean Average Precision at 100):
# This metric gives a broader view of performance across the top 100 results.
# It rewards models for retrieving many relevant documents, regardless of their
# exact order (as long as they are in the top 100). A higher MAP means better
# overall retrieval of relevant items.
