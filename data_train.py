import os
import random
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from transformers.trainer_utils import set_seed

# Initialize DDP for 2 GPUs
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl", world_size=2, rank=local_rank)

# Set seed for reproducibility
set_seed(42)

# Load dataset
directory = "/scratch/el3136/climate_text_dataset"
txt_files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".txt")])
random.shuffle(txt_files)
split_index = int(len(txt_files) * 0.9)
train_files, test_files = txt_files[:split_index], txt_files[split_index:]

train_dataset = load_dataset("text", data_files={"train": train_files})["train"]
eval_dataset = load_dataset("text", data_files={"test": test_files})["test"]

# Load tokenizer
model_name = "/scratch/el3136/Llama3.2-3B/"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

def tokenize_function(example):
    tokenized = tokenizer(
        example["text"], truncation=True, padding="max_length", max_length=512
    )
    tokenized["labels"] = tokenized["input_ids"]
    return tokenized

train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Load quantized model
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config).cuda(local_rank)
model.gradient_checkpointing_enable()

# Apply LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
    bias="none",
)
model = get_peft_model(model, lora_config)

for param in model.parameters():
    if param.dtype in [torch.float16, torch.float32, torch.float64]:
        param.requires_grad = True

# Wrap model with DDP
model = DistributedDataParallel(model, device_ids=[local_rank])

# Training arguments
training_args = TrainingArguments(
    output_dir="/scratch/el3136/data-finetuned-llama",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    optim="adamw_torch",
    fp16=True,
    logging_steps=10,
    save_steps=500,
    save_total_limit=3,
    evaluation_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=True,
    remove_unused_columns=False,
    ddp_find_unused_parameters=False,  # Required for PEFT + DDP
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# Train
trainer.train()
