import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    GenerationConfig,
    Trainer,
    TrainingArguments,
)

from eco.dataset import dataset_classes
from eco.utils import load_yaml

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, required=True)
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--num_epochs", type=float, default=5)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--loss", type=str, required=True)
parser.add_argument("--epsilon1", type=float, default=None)
parser.add_argument("--epsilon2", type=float, default=None)
parser.add_argument("--epsilon3", type=float, default=None)
args = parser.parse_args()

dataset_name = args.dataset_name
model_name = args.model_name
model_path = args.model_path
max_length = 256
batch_size = args.batch_size
num_epochs = args.num_epochs
learning_rate = args.lr

model_config = load_yaml(f"config/model_config/{model_name}.yaml")


class DistillKL(nn.Module):
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_s = p_s.view(-1, p_s.size(-1))
        p_t = F.softmax(y_t / self.T, dim=1)
        p_t = p_t.view(-1, p_t.size(-1))
        loss = F.kl_div(p_s, p_t, reduction="batchmean") * (self.T**2) / y_s.shape[0]
        return loss


class CustomDataCollator(DataCollatorWithPadding):
    def __init__(self, tokenizer, padding=True):
        super().__init__(tokenizer, padding=padding)

    def __call__(self, features):
        retain_features = []
        forget_features = []
        random_features = []

        # Separate features based on type
        for feature in features:
            retain_features.append(
                {
                    "input_ids": feature["retain_input_ids"],
                    "attention_mask": feature["retain_attention_mask"],
                }
            )
            forget_features.append(
                {
                    "input_ids": feature["forget_input_ids"],
                    "attention_mask": feature["forget_attention_mask"],
                }
            )
            random_features.append(
                {
                    "input_ids": feature["random_input_ids"],
                    "attention_mask": feature["random_attention_mask"],
                }
            )

        retain_batch = super().__call__(retain_features)
        forget_batch = super().__call__(forget_features)
        random_batch = super().__call__(random_features)

        batch = {
            "retain_input_ids": retain_batch["input_ids"],
            "retain_attention_mask": retain_batch["attention_mask"],
            "forget_input_ids": forget_batch["input_ids"],
            "forget_attention_mask": forget_batch["attention_mask"],
            "random_input_ids": random_batch["input_ids"],
            "random_attention_mask": random_batch["attention_mask"],
        }

        return batch


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.loss = kwargs.pop("loss")
        self.reference_model = kwargs.pop("reference_model", None)
        self.epsilon1 = kwargs.pop("epsilon1", None)
        self.epsilon2 = kwargs.pop("epsilon2", None)
        self.epsilon3 = kwargs.pop("epsilon3", None)
        self.scrub_max_epochs = kwargs.pop("scrub_max_epochs", None)
        super().__init__(*args, **kwargs)

        if self.loss == "scrub":
            self.distill = DistillKL(T=4.0)
            self.has_printed = False
            self.previous_step = None

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.loss == "ft":
            inputs = {
                "input_ids": inputs["retain_input_ids"],
                "attention_mask": inputs["retain_attention_mask"],
                "labels": inputs["retain_input_ids"],
            }
            outputs = model(**inputs)
            loss = outputs.loss
        elif self.loss == "ga":
            inputs = {
                "input_ids": inputs["forget_input_ids"],
                "attention_mask": inputs["forget_attention_mask"],
                "labels": inputs["forget_input_ids"],
            }
            outputs = model(**inputs)
            loss = outputs.loss * -1
        elif self.loss == "gd":
            retain_inputs = {
                "input_ids": inputs["retain_input_ids"],
                "attention_mask": inputs["retain_attention_mask"],
                "labels": inputs["retain_input_ids"],
            }
            forget_inputs = {
                "input_ids": inputs["forget_input_ids"],
                "attention_mask": inputs["forget_attention_mask"],
                "labels": inputs["forget_input_ids"],
            }
            retain_outputs = model(**retain_inputs)
            forget_outputs = model(**forget_inputs)
            loss = -1 * forget_outputs.loss + retain_outputs.loss
            outputs = forget_outputs
        elif self.loss == "kl":
            assert (
                self.reference_model is not None
            ), "Reference model must be provided for kl loss"
            retain_inputs = {
                "input_ids": inputs["retain_input_ids"],
                "attention_mask": inputs["retain_attention_mask"],
                "labels": inputs["retain_input_ids"],
            }
            forget_inputs = {
                "input_ids": inputs["forget_input_ids"],
                "attention_mask": inputs["forget_attention_mask"],
                "labels": inputs["forget_input_ids"],
            }
            retain_outputs = model(**retain_inputs)
            forget_outputs = model(**forget_inputs)

            with torch.no_grad():
                reference_outputs = self.reference_model(**retain_inputs)

            retain_dist = F.log_softmax(retain_outputs.logits, dim=-1)
            retain_dist = retain_dist.view(-1, retain_dist.size(-1))
            reference_dist = F.log_softmax(reference_outputs.logits, dim=-1)
            reference_dist = reference_dist.view(-1, reference_dist.size(-1))
            kl_loss = F.kl_div(
                retain_dist, reference_dist, reduction="batchmean", log_target=True
            )

            loss = -1 * forget_outputs.loss + kl_loss
            outputs = forget_outputs
        elif self.loss == "rd":
            retain_inputs = {
                "input_ids": inputs["retain_input_ids"],
                "attention_mask": inputs["retain_attention_mask"],
                "labels": inputs["retain_input_ids"],
            }
            random_inputs = {
                "input_ids": inputs["random_input_ids"],
                "attention_mask": inputs["random_attention_mask"],
                "labels": inputs["random_input_ids"],
            }
            retain_outputs = model(**retain_inputs)
            random_outputs = model(**random_inputs)
            loss = retain_outputs.loss + random_outputs.loss
            outputs = retain_outputs
        elif self.loss == "llmu":
            assert (
                self.reference_model is not None
            ), "Reference model must be provided for llmu loss"
            assert (
                self.epsilon1 is not None
                and self.epsilon2 is not None
                and self.epsilon3 is not None
            ), "Epsilon values must be provided for llmu loss"
            normal_inputs = {
                "input_ids": inputs["retain_input_ids"],
                "attention_mask": inputs["retain_attention_mask"],
                "labels": inputs["retain_input_ids"],
            }
            forget_inputs = {
                "input_ids": inputs["forget_input_ids"],
                "attention_mask": inputs["forget_attention_mask"],
                "labels": inputs["forget_input_ids"],
            }
            random_inputs = {
                "input_ids": inputs["random_input_ids"],
                "attention_mask": inputs["random_attention_mask"],
                "labels": inputs["random_input_ids"],
            }
            normal_outputs = model(**normal_inputs)
            forget_outputs = model(**forget_inputs)
            random_outputs = model(**random_inputs)

            with torch.no_grad():
                reference_outputs = self.reference_model(**normal_inputs)
            reference_dist = F.log_softmax(reference_outputs.logits, dim=-1)
            reference_dist = reference_dist.view(-1, reference_dist.size(-1))
            normal_dist = F.log_softmax(normal_outputs.logits, dim=-1)
            normal_dist = normal_dist.view(-1, normal_dist.size(-1))
            normal_loss = F.kl_div(
                normal_dist, reference_dist, reduction="batchmean", log_target=True
            )

            loss = (
                -1 * forget_outputs.loss * self.epsilon1
                + self.epsilon2 * random_outputs.loss
                + self.epsilon3 * normal_loss
            )
            outputs = forget_outputs
        elif self.loss == "scrub":
            alpha = self.epsilon1
            gamma = self.epsilon2
            current_epoch = self.state.epoch
            assert (
                self.reference_model is not None
            ), "Reference model must be provided for scrub loss"
            retain_inputs = {
                "input_ids": inputs["retain_input_ids"],
                "attention_mask": inputs["retain_attention_mask"],
                "labels": inputs["retain_input_ids"],
            }
            forget_inputs = {
                "input_ids": inputs["forget_input_ids"],
                "attention_mask": inputs["forget_attention_mask"],
                "labels": inputs["forget_input_ids"],
            }

            current_step = (
                "max"
                if current_epoch < self.scrub_max_epochs and int(current_epoch) % 2 == 0
                else "min"
            )

            if self.previous_step != current_step:
                self.previous_step = current_step
                print(f"SCRUB {current_step} step: {current_epoch}")

            # Max step
            if current_epoch < self.scrub_max_epochs and int(current_epoch) % 2 == 0:
                with torch.no_grad():
                    teacher_forget_outputs = self.reference_model(**forget_inputs)
                forget_outputs = model(**forget_inputs)
                forget_distill_loss = (
                    self.distill(forget_outputs.logits, teacher_forget_outputs.logits)
                    * -1
                )
                loss = forget_distill_loss
                outputs = forget_outputs
            # Min step
            else:
                retain_outputs = model(**retain_inputs)
                with torch.no_grad():
                    teacher_retain_outputs = self.reference_model(**retain_inputs)

                retain_distill_loss = self.distill(
                    retain_outputs.logits, teacher_retain_outputs.logits
                )
                loss = alpha * retain_distill_loss + gamma * retain_outputs.loss
                outputs = retain_outputs

        return (loss, outputs) if return_outputs else loss


tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if "qwen" in model_path.lower():
    tokenizer.padding_side = "left"
data_module = dataset_classes[dataset_name](tokenizer, max_length=256)
dataset = data_module.load_dataset_for_baseline_unlearn()
print(f"Dataset size: {len(dataset)}")


model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation=model_config["attn_implementation"],
    trust_remote_code=True,
)
reference_model = None
if args.loss in ["kl", "llmu", "scrub"]:
    reference_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=model_config["attn_implementation"],
        trust_remote_code=True,
        device_map="auto",
    )
scrub_max_epochs = None
if args.loss == "scrub":
    scrub_max_epochs = 4

model.generation_config = GenerationConfig()
print(f"Total parameters: {model.num_parameters()}")

data_collator = CustomDataCollator(tokenizer=tokenizer)
total_steps = int(
    (
        int(len(dataset) / batch_size / int(os.environ.get("WORLD_SIZE", 1)))
        + (len(dataset) % batch_size != 0)  # Add 1 if there's a remainder
    )
    * num_epochs
)
print(f"Total steps: {total_steps}")

save_dir = f"./{model_path}_{args.loss}"
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
    # Manually define save_steps for OLMo-1.7-7B-hf as the saving process
    # often stuck with the default value.
    save_steps=total_steps if model_name == "OLMo-1.7-7B-hf" else 1e8,
    remove_unused_columns=False,
    gradient_checkpointing=(
        True if args.loss in ["llmu", "kl", "rd", "gd", "scrub"] else False
    ),
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    loss=args.loss,
    reference_model=reference_model,
    epsilon1=args.epsilon1,
    epsilon2=args.epsilon2,
    epsilon3=args.epsilon3,
    scrub_max_epochs=scrub_max_epochs,
)

trainer.train()
if model_name != "OLMo-1.7-7B-hf":
    trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)
