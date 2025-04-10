import os
import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader

# Load testing text file paths
with open("/scratch/el3136/BDML-1/test_txt_files.txt", "r") as f:
    test_txt_files = f.read().splitlines()

# ================== EVALUATION STEP ==================

# Load model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "/scratch/el3136/your-finetuned-llama"
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Custom Dataset for evaluation
class TextFileDataset(Dataset):
    def __init__(self, file_paths, tokenizer, max_length=512):
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Open file on-the-fly
        file_path = self.file_paths[idx]
        with open(file_path, "r") as f:
            text = f.read()

        # Tokenize text
        tokenized = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        tokenized["labels"] = tokenized["input_ids"]
        return tokenized

# Create eval dataset and dataloader
eval_dataset = TextFileDataset(test_txt_files, tokenizer)
eval_dataloader = DataLoader(eval_dataset, batch_size=1)  # One file at a time

def compute_perplexity(input_ids):
    """Computes perplexity of a given input."""
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss.item()  # get the loss value
    return math.exp(loss)

# Evaluate the model on the test set
total_perplexity = 0.0
num_samples = 0

for batch in eval_dataloader:
    input_ids = batch["input_ids"].squeeze(1)  # Remove batch dimension (since batch_size=1)
    perplexity = compute_perplexity(input_ids)
    total_perplexity += perplexity
    num_samples += 1

average_perplexity = total_perplexity / num_samples
print(f"Average Perplexity: {average_perplexity:.2f}")
