# ==============================================================================
#           Training a Small ModernBERT for Sentence Embeddings
# ==============================================================================
#
# PURPOSE:
# This script (train_small_modernbert.py) serves as a complete example for fine-tuning a custom, small-scale
# ModernBERT model to generate high-quality sentence embeddings. It follows the
# modern training pipeline introduced in sentence-transformers v3.
#
# This file is heavily commented to serve as a learning resource for
# anyone new to training embedding models. Each step explains
# not just *what* we're doing, but *why* we're doing it.
#
# BEFORE YOU RUN:
# 1. Make sure you've already created the blank model using the
#    `build_and_save_small_modernbert.py` script.
# 2. Install the necessary libraries:
#    pip install sentence-transformers datasets accelerate
#
# `accelerate` is a library from Hugging Face that helps optimize training
# across different hardware (like GPUs) automatically.
#
# ==============================================================================

import torch

if torch.cuda.is_available():
    print("GPU (CUDA) is available for PyTorch.")
    print(f"Number of available GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU device name: {torch.cuda.get_device_name(0)}") # Get name of first GPU
else:
    print("GPU (CUDA) is not available for PyTorch. PyTorch will run on CPU.")
    
from sentence_transformers import (
    models,
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData, # For creating a nice model card on the Hub
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers # For optimizing our loss function
from sentence_transformers.evaluation import (
    TripletEvaluator,
    EmbeddingSimilarityEvaluator,
    SequentialEvaluator, # To combine multiple evaluators
    SimilarityFunction,
)
from datasets import load_dataset

# --- Step 1: Initialize Our Model ---
# Here, we're not starting with a pre-trained model from the internet. Instead,
# we're loading the custom, blank ModernBERT architecture that we designed and saved
# in the previous step. This gives us full control over the model's size.

# I've set the path to where we saved our blank model. Make sure this is correct.
model_path = "./ModernBERT-small"
print(f"INFO: Loading our custom blank model architecture from: {model_path}")

# A SentenceTransformer model is built from modules, like Lego bricks.
# Our model will have two essential parts:
# 1. A Transformer layer: This is our ModernBERT model. It reads the text
#    and outputs embeddings for every single token.
word_embedding_model = models.Transformer(model_path)

# 2. A Pooling layer: The transformer gives us many token embeddings, but we need
#    a single vector for the whole sentence. The pooling layer handles this by
#    averaging the token embeddings (mean pooling). This is the most common approach.
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(), # Gets the hidden size (e.g., 256)
    pooling_mode='mean'
)

# Now, we assemble these two modules into a final SentenceTransformer model.
# The `SentenceTransformerModelCardData` is good practice; it helps auto-generate
# a README file if we ever decide to upload our model to the Hugging Face Hub.
model = SentenceTransformer(
    modules=[word_embedding_model, pooling_model],
    model_card_data=SentenceTransformerModelCardData(
        language="en",
        license="apache-2.0",
        model_name="Small ModernBERT for Sentence Similarity",
    )
)
print("SUCCESS: Blank ModernBERT model loaded into a SentenceTransformer wrapper.")
print(model)


# --- Step 2: Prepare the Datasets ---
# A model is only as good as its data. We need a well-formatted dataset to teach
# our model what "similarity" means.
#
# For this project, I chose `sentence-transformers/all-nli`. It contains over 1 million
# sentence triplets in the format (anchor, positive, negative).
#  - 'anchor': An original sentence.
#  - 'positive': A sentence that is semantically similar to the anchor.
#  - 'negative': A sentence that is unrelated to the anchor.
# This format is perfect for "contrastive learning".
dataset_name = "sentence-transformers/all-nli"
print(f"\nINFO: Loading dataset '{dataset_name}' for training...")

# This is the dataset our model will learn from.
train_dataset = load_dataset(dataset_name, "triplet", split="train[:50000]")

# We'll also load a separate 'development' or 'validation' set. The model never
# trains on this data; we only use it to check how well the model is learning.
eval_dataset_nli = load_dataset(dataset_name, "triplet", split="dev[:1000]")

# To get a more robust measure of performance, I'm also loading the famous
# STS benchmark (STSb). It contains pairs of sentences with a human-rated
# similarity score (from 0 to 5). This helps us see if our model's understanding
# of similarity matches that of a human.
print("INFO: Loading STS-benchmark dataset for a second evaluation metric...")
eval_dataset_stsb = load_dataset("sentence-transformers/stsb", split="validation")


# --- Step 3: Define the Loss Function ---
# The loss function is the "teacher" that tells the model how to improve.
# Since our data is in (anchor, positive, negative) format, the `MultipleNegativesRankingLoss`
# is an excellent choice.
#
# Its goal is simple: make the distance between (anchor, positive) SMALLER than the
# distance between (anchor, negative). It's a very effective loss for training
# embedding models.
print("\nINFO: Defining the loss function: MultipleNegativesRankingLoss.")
loss = MultipleNegativesRankingLoss(model)


# --- Step 4: Configure Training Arguments ---
# These are the hyperparameters that control the training process. Think of them as
# the settings on a machine.
output_dir = "output/training-small-modernbert"
print(f"\nINFO: Training arguments configured. Checkpoints will be saved to: {output_dir}")

args = SentenceTransformerTrainingArguments(
    # Required: Where to save checkpoints.
    output_dir=output_dir,

    # --- Key Training Parameters ---
    # After reviewing the literature and based on the size of our dataset,
    # one epoch is a strong and safe starting point. More epochs can lead to overfitting.
    num_train_epochs=1,
    per_device_train_batch_size=32, # How many examples to process at once. Adjust based on your GPU's VRAM.
    learning_rate=2e-5, # A standard, effective learning rate for fine-tuning transformers.
    warmup_ratio=0.1, # For the first 10% of training, the learning rate will slowly ramp up. This helps stabilize training.
    fp16=True, # Use 16-bit floating point precision. Makes training faster and uses less memory on compatible GPUs.

    # This is a recommendation from the docs for our specific loss function.
    # It ensures each training batch contains unique sentences, which makes the
    # "in-batch negatives" technique more effective.
    batch_sampler=BatchSamplers.NO_DUPLICATES,

    # --- Evaluation and Saving Strategy ---
    eval_strategy="steps", # How often to evaluate. We'll do it based on a number of steps.
    eval_steps=500,       # Run the evaluation every 500 training steps.
    save_strategy="steps", # How often to save a model checkpoint.
    save_steps=500,
    save_total_limit=2,    # To save disk space, only keep the 2 most recent checkpoints.

    # This is a crucial setting. At the end of training, it will automatically
    # load the checkpoint that had the best performance on our evaluation set.
    load_best_model_at_end=True,
    metric_for_best_model="sts-dev_spearman_cosine", # The STS benchmark is the most common and respected way to report the performance of sentence embedding models.
    # The available evaluation metrics are: ['eval_all-nli-dev_cosine_accuracy', 'eval_sts-dev_pearson_cosine', 'eval_sts-dev_spearman_cosine', 'eval_sequential_score', 'eval_runtime', 'eval_samples_per_second', 'eval_steps_per_second', 'epoch']

    # --- Logging and Reporting ---
    logging_steps=100, # How often to print the training loss to the console.
    run_name="small-modernbert-all-nli", # A name for the run, useful if you use Weights & Biases.
)


# --- Step 5: Set Up the Evaluators ---
# While the training loss tells us if the model is learning, evaluators give us
# interpretable, real-world metrics. I've set up two:

print("\nINFO: Setting up evaluators for validation...")
# Evaluator 1: Checks our main training objective on the NLI dev set.
# It measures "triplet accuracy": what percentage of the time is the model
# correctly identifying the 'positive' sentence as more similar than the 'negative'?
nli_evaluator = TripletEvaluator(
    anchors=eval_dataset_nli["anchor"],
    positives=eval_dataset_nli["positive"],
    negatives=eval_dataset_nli["negative"],
    name="all-nli-dev", # A label for the output logs
)

# Evaluator 2: Checks performance on the STSb dataset.
# This measures the "Spearman correlation" between our model's similarity scores
# and the human-annotated scores. A higher correlation means our model's
# sense of similarity is closer to a human's.
stsb_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=eval_dataset_stsb["sentence1"],
    sentences2=eval_dataset_stsb["sentence2"],
    scores=eval_dataset_stsb["score"],
    main_similarity=SimilarityFunction.COSINE,
    name="sts-dev",
)

# The `SequentialEvaluator` is a handy tool that runs all our evaluators in order
# during each evaluation step. This gives us a comprehensive view of performance.
evaluator = SequentialEvaluator([nli_evaluator, stsb_evaluator])


# --- Step 6: Initialize and Start the Trainer ---
# This is where we bring everything together: the model, arguments, datasets,
# loss function, and evaluators. The `SentenceTransformerTrainer` handles all the
# complexity of the training loop for us.
print("\nINFO: Initializing the SentenceTransformerTrainer...")
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=loss,
    evaluator=evaluator,
)

print("\n🚀🚀🚀 STARTING TRAINING 🚀🚀🚀")
trainer.train()
print("\n🏁🏁🏁 TRAINING COMPLETE 🏁🏁🏁")


# --- Step 7: Save the Final, Trained Model ---
# Thanks to `load_best_model_at_end=True`, the `model` object in our script
# is now the version that performed best on our validation set. We can now
# save this final, high-quality model for future use.
final_model_path = f"{output_dir}/final-best"
print(f"\nINFO: Saving the final, best-performing model to: {final_model_path}")
model.save_pretrained(final_model_path)

print(f"\n✅ All done! Your newly trained sentence-embedding model is ready at '{final_model_path}'.")