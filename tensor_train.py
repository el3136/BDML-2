import os
import torch
import torch.distributed as dist
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from torch.distributed.tensor.parallel import parallelize_module
import random

# Argument Parsing for Tensor Parallel
rank  = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl", world_size=2, rank=rank)

# Define the directory containing txts
directory = "/scratch/el3136/climate_text_dataset"
txt_files = sorted(
    [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".txt")]
)

# Shuffle and split into 90% training and 10% testing
random.seed(42)
random.shuffle(txt_files)
split_index = int(len(txt_files) * 0.9)
train_sets, test_sets = txt_files[:split_index], txt_files[split_index:]

# Save paths
with open("/scratch/el3136/BDML-1/train_txt_files.txt", "w") as f:
    f.write("\n".join(train_sets))
with open("/scratch/el3136/BDML-1/test_txt_files.txt", "w") as f:
    f.write("\n".join(test_sets))

# Load paths
with open("/scratch/el3136/BDML-1/train_txt_files.txt", "r") as f:
    train_txt_files = f.read().splitlines()
with open("/scratch/el3136/BDML-1/test_txt_files.txt", "r") as f:
    test_txt_files = f.read().splitlines()

# Load datasets
train_dataset = load_dataset("text", data_files={"train": train_txt_files})["train"]
eval_dataset = load_dataset("text", data_files={"test": test_txt_files})["test"]

# Load model and tokenizer
model_name = "/scratch/el3136/Llama3.2-3B/"
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config).cuda(rank)

# Tensor parallelism
model = parallelize_module(model, parallel_mode="column", devices=[0, 1])

# Tokenizer setup
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

def tokenize_function(example):
    tokenized = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    tokenized["labels"] = tokenized["input_ids"]
    return tokenized

train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

model.gradient_checkpointing_enable()

# LoRA config
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
    bias="none"
)
lora_model = get_peft_model(model, lora_config)

for param in lora_model.parameters():
    if param.dtype in [torch.float32, torch.float16, torch.float64]:
        param.requires_grad = True

lora_model.print_trainable_parameters()

# Training arguments
training_args = TrainingArguments(
    output_dir="/scratch/el3136/tensor-finetuned-llama",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    optim="adamw_torch",
    fp16=True,
    bf16=False,
    logging_steps=10,
    save_steps=500,
    save_total_limit=3,
    evaluation_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=True,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

trainer.train()
