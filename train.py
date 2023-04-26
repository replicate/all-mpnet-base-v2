import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, InputExample, losses
from datasets import load_dataset

# Load the MNLI dataset from Hugging Face Datasets
mnli_dataset = load_dataset("multi_nli")

# Preprocess the dataset
def preprocess(example):
    return {
        "premise": example["premise"],
        "hypothesis": example["hypothesis"],
        "label": example["label"],
    }

mnli_train = mnli_dataset["train"].map(preprocess)
mnli_val = mnli_dataset["validation_matched"].map(preprocess)

# Create InputExamples for SentenceTransformer
train_examples = [
    InputExample(texts=[x["premise"], x["hypothesis"]], label=x["label"])
    for x in mnli_train
]

val_examples = [
    InputExample(texts=[x["premise"], x["hypothesis"]], label=x["label"])
    for x in mnli_val
]

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
model = AutoModel.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")

# Create a SentenceTransformer instance
sentence_transformer = SentenceTransformer(modules=[tokenizer, model])

# DataLoader for the training and validation datasets
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
val_dataloader = DataLoader(val_examples, shuffle=False, batch_size=32)

# Use CosineSimilarityLoss
train_loss = losses.CosineSimilarityLoss(sentence_transformer)

# Fine-tune the model
sentence_transformer.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    warmup_steps=100,
    evaluator=val_dataloader,
    evaluation_steps=500,
    output_path="fine-tuned-model",
)
