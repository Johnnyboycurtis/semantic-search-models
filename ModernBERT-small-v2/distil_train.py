"""
This file contains an example how to make a SentenceTransformer model faster and lighter.
"""

import logging
import traceback
from datetime import datetime

import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from sklearn.decomposition import PCA

from sentence_transformers import LoggingHandler, SentenceTransformer, evaluation, losses, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import NanoBEIREvaluator


retrieval_evaluator = NanoBEIREvaluator(
    dataset_names=["MSMARCO", "HotpotQA"],
    batch_size=32
)

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
#### /print debug information to stdout


# Teacher Model: Model we want to distill to a smaller model
teacher_model_name = "Alibaba-NLP/gte-modernbert-base"
teacher_model = SentenceTransformer(teacher_model_name,
    model_kwargs={
        "attn_implementation": "flash_attention_2",
        "device_map": "cuda",
    }
)

output_dir = "output/model-distillation-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# We will train a small model like TinyBERT to imitate the teacher.
STAGE_1_PATH = "./ModernBERT-small-2/pre-trained/final_model"
logging.info(f"Loading Stage 1 Model from: {STAGE_1_PATH}")

student_model = SentenceTransformer(
    STAGE_1_PATH,
    model_kwargs={
        "attn_implementation": "flash_attention_2",
        "device_map": "cuda",
    }
)

inference_batch_size = 64
train_batch_size = 64

def deduplicate(dataset):
    df = pd.DataFrame(dataset)
    df = df.drop_duplicates()
    return Dataset.from_pandas(df, preserve_index=False)

# ---------------------------------------------------------
# 1. Load AllNLI
# ---------------------------------------------------------
logging.info("Load the AllNLI dataset")
nli_train_dataset = load_dataset("sentence-transformers/all-nli", "pair-score", split="train")
nli_eval_dataset = load_dataset("sentence-transformers/all-nli", "pair-score", split="dev")

def combine_sentences(batch):
    return {"sentence": batch["sentence1"] + batch["sentence2"]}

nli_train_dataset = nli_train_dataset.map(
    combine_sentences, batched=True, remove_columns=nli_train_dataset.column_names
)
nli_eval_dataset = nli_eval_dataset.map(combine_sentences, batched=True, remove_columns=nli_eval_dataset.column_names)

nli_train_dataset = deduplicate(nli_train_dataset)
nli_eval_dataset = deduplicate(nli_eval_dataset)
logging.info(nli_train_dataset)

# ---------------------------------------------------------
# 2. Load Wikipedia
# ---------------------------------------------------------
logging.info("Load the Wikipedia dataset")
wikipedia_train_dataset = load_dataset("sentence-transformers/wikipedia-en-sentences", split="train")
# Take 5000 random sentences from the Wikipedia dataset for evaluation
wikipedia_train_dataset_dict = wikipedia_train_dataset.train_test_split(test_size=5000)
wikipedia_train_dataset = wikipedia_train_dataset_dict["train"]
wikipedia_eval_dataset = wikipedia_train_dataset_dict["test"]
logging.info(wikipedia_train_dataset)

# ---------------------------------------------------------
# 3. Load and Prep Philosophy Dataset (NEW ADDITION)
# ---------------------------------------------------------
logging.info("Load the Philosophy dataset")
# TODO: Replace 'your-org/philosophy-dataset' with the actual path or variable
philosophy_ds = load_dataset("johnnyboycurtis/Philosophical-STS-Text-Pairs", split="train")

# --------------------------------------------------------------------------

# Filter: Keep only rows where score > 0.7
logging.info(f"Original Philosophy size: {len(philosophy_ds)}")
# philosophy_ds = philosophy_ds.filter(lambda x: x['llm_score'] > 0.7)
# logging.info(f"Filtered Philosophy size (>0.7): {len(philosophy_ds)}")

# We flatten the pairs. Since this is distillation (MSELoss), we don't need pairs (Anchor/Positive).
# We just need raw sentences to pass through the teacher to get embeddings.
def flatten_philosophy(batch):
    # This combines text1 list and text2 list into one long list of 'sentence'
    return {"sentence": batch["text1"] + batch["text2"]}

philosophy_train_dataset = philosophy_ds.map(
    flatten_philosophy, 
    batched=True, 
    remove_columns=philosophy_ds.column_names # This removes text1, text2, and llm_score
)

philosophy_train_dataset = deduplicate(philosophy_train_dataset)
logging.info(f"Final Philosophy Dataset for Distillation: {philosophy_train_dataset}")

# ---------------------------------------------------------
# 4. Concatenate and Prep
# ---------------------------------------------------------

logging.info("Load the STSB dataset")
stsb_eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")
stsb_test_dataset = load_dataset("sentence-transformers/stsb", split="test")

# Concatenate NLI, Wikipedia, AND Philosophy for training
train_dataset: Dataset = concatenate_datasets([
    nli_train_dataset, 
    wikipedia_train_dataset,
    philosophy_train_dataset # <--- Added here
])

# Create a relatively small dataset for evaluation
eval_dataset: Dataset = concatenate_datasets(
    [nli_eval_dataset.select(range(5000)), wikipedia_eval_dataset.select(range(5000))]
)

# Create an STSB evaluator
dev_evaluator_stsb = EmbeddingSimilarityEvaluator(
    sentences1=list(stsb_eval_dataset["sentence1"]),
    sentences2=list(stsb_eval_dataset["sentence2"]),
    scores=list(stsb_eval_dataset["score"]),
    main_similarity=SimilarityFunction.COSINE,
    name="sts-dev",
)
logging.info("Teacher Performance")
dev_evaluator_stsb(teacher_model)

# Student model has fewer dimensions. Compute PCA for the teacher to reduce the dimensions
if student_model.get_sentence_embedding_dimension() < teacher_model.get_sentence_embedding_dimension():
    logging.info("Student model has fewer dimensions than the teacher. Compute PCA for down projection")
    
    # We take a sample from NLI and Wiki for PCA calculation (usually sufficient)
    pca_sentences = nli_train_dataset[:20000]["sentence"] + wikipedia_train_dataset[:20000]["sentence"]
    pca_embeddings = teacher_model.encode(pca_sentences, convert_to_numpy=True)
    pca = PCA(n_components=student_model.get_sentence_embedding_dimension())
    pca.fit(pca_embeddings)

    # Add Dense layer to teacher that projects the embeddings down to the student embedding size
    dense = models.Dense(
        in_features=teacher_model.get_sentence_embedding_dimension(),
        out_features=student_model.get_sentence_embedding_dimension(),
        bias=False,
        activation_function=torch.nn.Identity(),
    )
    dense.linear.weight = torch.nn.Parameter(torch.tensor(pca.components_))
    teacher_model.add_module("dense", dense)

    logging.info(f"Teacher Performance with {teacher_model.get_sentence_embedding_dimension()} dimensions:")
    dev_evaluator_stsb(teacher_model)


# Use the teacher model to get the gold embeddings
def map_embeddings(batch):
    return {
        "label": teacher_model.encode(
            batch["sentence"], batch_size=inference_batch_size, show_progress_bar=False
        ).tolist()
    }


train_dataset = train_dataset.shuffle(123).select(range(1000000))
train_dataset = train_dataset.map(map_embeddings, batched=True, batch_size=50000)
# Optionally, save the dataset to disk to speed up future runs
train_dataset.save_to_disk("datasets/distillation_train_dataset")

eval_dataset = eval_dataset.map(map_embeddings, batched=True, batch_size=50000)

train_loss = losses.MSELoss(model=student_model)

# We create an evaluator, that measure the Mean Squared Error (MSE) between the teacher and the student embeddings
eval_sentences = list(eval_dataset["sentence"])
dev_evaluator_mse = evaluation.MSEEvaluator(eval_sentences, eval_sentences, teacher_model=teacher_model)
dev_evaluator = evaluation.SequentialEvaluator([dev_evaluator_stsb, dev_evaluator_mse, retrieval_evaluator])

# Define the training arguments
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=output_dir,
    # Optional training parameters:
    num_train_epochs=3,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=train_batch_size,
    warmup_ratio=0.1,
    fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=True,  # Set to True if you have a GPU that supports BF16
    metric_for_best_model="eval_sts-dev_spearman_cosine",
    load_best_model_at_end=True,
    learning_rate=1e-4,
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=4,
    logging_steps=100,
    run_name="distillation-layer-reduction",
)

# Create the trainer & start training
trainer = SentenceTransformerTrainer(
    model=student_model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=train_loss,
    evaluator=dev_evaluator,
)
trainer.train()

# Evaluate the model performance on the STS Benchmark test dataset
test_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=list(stsb_test_dataset["sentence1"]),
    sentences2=list(stsb_test_dataset["sentence2"]),
    scores=list(stsb_test_dataset["score"]),
    main_similarity=SimilarityFunction.COSINE,
    name="sts-test",
)
logging.info(test_evaluator(student_model))

trainer.train()

logging.info(test_evaluator(student_model))

# Save the trained & evaluated model locally
final_output_dir = f"{output_dir}/final"
student_model.save(final_output_dir)





res = retrieval_evaluator(student_model)


