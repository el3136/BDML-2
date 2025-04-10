import os
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
)
from datasets import Dataset, DatasetDict
from torch.utils.data import Dataset as TorchDataset
import random
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize the distributed environment
dist.init_process_group("nccl", rank=int(os.environ['LOCAL_RANK']), world_size=2)

# Define the directory containing txts
directory = "/scratch/el3136/climate_text_dataset"
txt_files = sorted(
    [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".txt")]
)

# Shuffle and split into 90% training and 10% testing
random.seed(42)  # Ensures consistent splits
random.shuffle(txt_files)
split_index = int(len(txt_files) * 0.9)
train_sets, test_sets = txt_files[:split_index], txt_files[split_index:]

# Save the list of generated text file paths
with open("/scratch/el3136/BDML-1/train_txt_files.txt", "w") as f:
    f.write("\n".join(train_sets))
with open("/scratch/el3136/BDML-1/test_txt_files.txt", "w") as f:
    f.write("\n".join(test_sets))

# Custom Dataset class to load text files efficiently
class TextFileDataset(TorchDataset):
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
        )
        tokenized["labels"] = tokenized["input_ids"]
        return tokenized

# Load the tokenizer and model
model_name = "meta-llama/Llama-3B"  # Use Hugging Face's Llama-3B model

# Enable 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Load quantized model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Ensure tokenizer padding side for causal LM
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Create datasets with the custom dataset class
train_dataset = TextFileDataset(train_sets, tokenizer)
eval_dataset = TextFileDataset(test_sets, tokenizer)

# ================== DISTRIBUTED DATA PARALLELISM (DDP) ==================

# Set the model to the appropriate device for the rank
device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
model = model.to(device)

# Apply DistributedDataParallel (DDP)
model = DDP(model, device_ids=[int(os.environ['LOCAL_RANK'])])

# Define training arguments
training_args = TrainingArguments(
    output_dir="/scratch/el3136/your-finetuned-llama",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    optim="adamw_torch",
    fp16=False,           # Set fp16=False as per your requirement
    bf16=True,            # Enable bf16 for hardware optimization
    logging_steps=10,
    save_steps=500,
    save_total_limit=3,
    evaluation_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=True,
    remove_unused_columns=False,  # prevent dropping required fields
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# Start training
trainer.train()
