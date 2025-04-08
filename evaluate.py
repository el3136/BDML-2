import os
import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Load testing text file paths
with open("/scratch/el3136/BDML-1/test_txt_files.txt", "r") as f:
    test_txt_files = f.read().splitlines()

eval_dataset = load_dataset("text", data_files={"test": test_txt_files})["test"]

# ================== EVALUATION STEP ==================

# Load model and tokenizer once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "/scratch/el3136/data-finetuned-llama"
model_path = "/scratch/el3136/tensor-finetuned-llama"
model_path = "/scratch/el3136/pipeline-finetuned-llama"
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

def compute_perplexity(text):
    """Computes perplexity of a given text."""
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**tokens, labels=tokens['input_ids'])
        loss = outputs.loss.item()
    return math.exp(loss)

# Evaluate on a sample text from the test set
sample = next(iter(eval_dataset))
eval_text = sample["text"] if isinstance(sample, dict) else sample
perplexity = compute_perplexity(eval_text)

print(f"Perplexity: {perplexity:.2f}")
