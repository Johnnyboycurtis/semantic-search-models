# First, ensure you have the necessary libraries installed
# pip install sentence-transformers datasets accelerate

import os
from sentence_transformers import (
    models,
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
from datasets import load_dataset

# --- 1. Load the Model ---
# This is the most crucial step for your project. Instead of starting from a
# pre-trained model name like "microsoft/mpnet-base", we load the custom
# ModernBERT model you already designed and saved.

model_path = "./ModernBERT-small" # Ensure this path is correct!

# Create a SentenceTransformer model from your saved, blank ModernBERT.
# This follows the documentation's advice: a Transformer layer followed by a Pooling layer.
print(f"Loading blank model from: {model_path}")
word_embedding_model = models.Transformer(model_path)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

print("Blank ModernBERT model loaded into SentenceTransformer:")
print(model)


# --- 2. Prepare the Dataset ---
# As per the documentation, we'll load a dataset from the Hugging Face Hub.
# The 'all-nli' dataset is perfect because it comes in various formats.
# We'll use the 'triplet' format, which is ideal for contrastive learning.
dataset_name = "sentence-transformers/all-nli"
print(f"\nLoading dataset: {dataset_name}")

# For a quick demonstration, we'll only use a small subset of the data.
# In a real project, you would use the full dataset.
train_dataset = load_dataset(dataset_name, "triplet", split="train")
eval_dataset = load_dataset(dataset_name, "triplet", split="dev")

print("\nTrain dataset sample:")
print(train_dataset[0])
# Expected output: {'anchor': '...', 'positive': '...', 'negative': '...'}


# --- 3. Choose a Loss Function ---
# The loss function's choice depends on the dataset format. Since we have
# (anchor, positive, negative) triplets, MultipleNegativesRankingLoss is an excellent choice.
# It pushes the (anchor, positive) pair to be closer than the (anchor, negative) pair.
print("\nDefining the loss function: MultipleNegativesRankingLoss")
loss = MultipleNegativesRankingLoss(model)


# --- 4. Define Training Arguments ---
# These arguments control the training process. We'll set a few key ones.
output_dir = "output/training-small-modernbert"
args = SentenceTransformerTrainingArguments(
    # Required parameter
    output_dir=output_dir,
    # Training parameters
    num_train_epochs=3,
    per_device_train_batch_size=32,  # Adjust based on your VRAM
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if your GPU can't handle FP16
    bf16=False,  # Set to True if your GPU supports BF16
    # Evaluation parameters
    eval_strategy="steps",
    eval_steps=500, # Evaluate every 500 steps
    # Saving parameters
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2, # Only keep the last 2 checkpoints
    # Logging
    logging_steps=50,
)

print(f"\nTraining arguments configured. Output will be saved to: {output_dir}")


# --- 5. Set up an Evaluator ---
# The evaluator provides more meaningful metrics than just the loss.
# A TripletEvaluator is perfect for our triplet dataset. It measures the
# percentage of triplets where the (anchor, positive) distance is smaller
# than the (anchor, negative) distance.
print("\nSetting up the TripletEvaluator for the validation set.")
evaluator = TripletEvaluator(
    anchors=eval_dataset["anchor"],
    positives=eval_dataset["positive"],
    negatives=eval_dataset["negative"],
    main_similarity_function=SimilarityFunction.COSINE,
    name="all-nli-dev",
)


# --- 6. Instantiate and Run the Trainer ---
# Now we bring all the components together in the SentenceTransformerTrainer.
print("\nInitializing the Trainer...")
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset, # The trainer uses this for loss calculation
    loss=loss,
    evaluator=evaluator, # The evaluator provides custom metrics
)

print("\nStarting training...")
trainer.train()

# --- 7. Save the Final Model ---
# After training, save the final, fine-tuned model.
final_model_path = f"{output_dir}/final"
print(f"\nTraining complete. Saving final model to: {final_model_path}")
model.save_pretrained(final_model_path)

# You can now load and use this model like any other SentenceTransformer model:
# trained_model = SentenceTransformer(final_model_path)