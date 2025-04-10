import os
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
import random

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

# Load training and testing text file paths
with open("/scratch/el3136/BDML-1/train_txt_files.txt", "r") as f:
    train_txt_files = f.read().splitlines()
with open("/scratch/el3136/BDML-1/test_txt_files.txt", "r") as f:
    test_txt_files = f.read().splitlines()

# Tells "trainer" that "train" is made up of the actual .txt files listed in train_txt_files
train_dataset = load_dataset("text", data_files={"train": train_txt_files})["train"]
eval_dataset = load_dataset("text", data_files={"test": test_txt_files})["test"]

# ================== TRAINING STEP ==================

# Define model path
model_name = "/scratch/el3136/Llama3.2-3B/"

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

# Tokenization function
def tokenize_function(example):
    # Here, we set `labels` to be the same as `input_ids` in causal LM
    tokenized = tokenizer(
        example["text"][0],
        truncation=False,
        # example["text"],
        # truncation=True,
        padding="max_length",
        max_length=512,
    )
    tokenized["labels"] = tokenized["input_ids"]
    return tokenized

# Apply tokenization
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Enable gradient checkpointing for memory optimization
model.gradient_checkpointing_enable()

# Define LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
    bias="none"
)

# Apply LoRA
lora_model = get_peft_model(model, lora_config)

# Ensure that only float-type parameters require gradients
for param in lora_model.parameters():
    if param.dtype in [torch.float32, torch.float16, torch.float64]:  # Check for float-type parameters
        param.requires_grad = True 

lora_model.print_trainable_parameters()

# Define training arguments
training_args = TrainingArguments(
    output_dir="/scratch/el3136/your-finetuned-llama",
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
    remove_unused_columns=False,  # prevent dropping required PeftModelForCausalLM fields
)

# Define Trainer
trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# Start training
trainer.train()
