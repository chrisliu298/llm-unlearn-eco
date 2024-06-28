import argparse
import os
from glob import glob

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GenerationConfig,
    Trainer,
    TrainingArguments,
)

from eco.dataset import dataset_classes

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, required=True)
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--num_epochs", type=float, default=5)
parser.add_argument("--lr", type=float, default=2e-5)
args = parser.parse_args()

dataset_name = args.dataset_name
model_name = args.model_name
max_length = 256
batch_size = args.batch_size
num_epochs = args.num_epochs
learning_rate = args.lr

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if "qwen" in model_name.lower():
    tokenizer.padding_side = "left"
data_module = dataset_classes[dataset_name](tokenizer, max_length=256)
dataset = data_module.load_dataset_for_train()
print(f"Dataset size: {len(dataset)}")

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
except:
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
model.generation_config = GenerationConfig()
print(f"Total parameters: {model.num_parameters()}")

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
total_steps = int(
    (
        int(len(dataset) / batch_size / int(os.environ.get("WORLD_SIZE", 1)))
        + (len(dataset) % batch_size != 0)  # Add 1 if there's a remainder
    )
    * num_epochs
)
print(f"Total steps: {total_steps}")

model_name_short = model_name.split("/")[-1]
save_dir = f"./{dataset_name}_{model_name_short}"
training_args = TrainingArguments(
    output_dir=save_dir,
    overwrite_output_dir=True,
    do_train=True,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=learning_rate,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    num_train_epochs=num_epochs,
    warmup_ratio=0.1,
    logging_strategy="steps",
    logging_first_step=True,
    logging_steps=total_steps // 50,
    bf16=True,
    optim="paged_adamw_32bit",
    report_to="none",
    save_only_model=True,
    save_steps=1e8,
)

trainer = Trainer(
    model=model, args=training_args, data_collator=data_collator, train_dataset=dataset
)

trainer.train()
trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)
