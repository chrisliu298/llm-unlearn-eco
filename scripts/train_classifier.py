import argparse

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from eco.dataset import (
    MMLUEconometrics,
    MMLUEconomics,
    MMLUJurisprudence,
    MMLULaw,
    MMLUMath,
    MMLUPhysics,
    MMLUWithoutEconomicsEconometrics,
    MMLUWithoutLawJurisprudence,
    MMLUWithoutPhysicsMath,
    dataset_classes,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_name", type=str, required=True, choices=dataset_classes.keys()
)
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--threshold", type=float, default=0.99)
parser.add_argument("--tofu_subset_name", type=str, default=None)
parser.add_argument("--mmlu_subset_name", type=str, default=None)
args = parser.parse_args()


def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    probs = torch.softmax(torch.tensor(logits), dim=-1).cpu().numpy()
    predictions = np.where(probs[:, 1] > args.threshold, 1, 0)
    accuracy = np.sum(predictions == labels) / len(labels)
    errors = np.sum(np.abs(labels - predictions))
    return {"errors": errors, "acc": accuracy}


# Load the tokenizer
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

if args.dataset == "tofu":
    assert args.tofu_subset_name is not None, "Must provide tofu_subset_name"
    data_module = dataset_classes[args.dataset_name](tokenizer, max_length=512)
    dataset = data_module.load_dataset_for_classification(args.tofu_subset_name)
elif args.dataset == "mmlu-subset":
    assert args.mmlu_subset_name is not None, "Must provide mmlu_subset_name"
    subset_to_cls = {
        "economics": [
            MMLUEconomics,
            MMLUEconometrics,
            MMLUWithoutEconomicsEconometrics,
        ],
        "physics": [MMLUPhysics, MMLUMath, MMLUWithoutPhysicsMath],
        "law": [MMLULaw, MMLUJurisprudence, MMLUWithoutLawJurisprudence],
    }
    data_module = dataset_classes[args.dataset_name]()
    dataset = data_module.load_dataset_for_classification(
        *subset_to_cls[args.mmlu_subset_name],
        use_val=True,
    )
else:
    data_module = dataset_classes[args.dataset_name]()
    dataset = data_module.load_dataset_for_classification(use_val=True)

num_class_0 = dataset["train"]["label"].count(0)
num_class_1 = dataset["train"]["label"].count(1)
class_weights = (
    torch.tensor(
        [
            (num_class_0 + num_class_1) / num_class_0,
            (num_class_0 + num_class_1) / num_class_1,
        ]
    )
    .float()
    .to(device)
)
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
print(f"Class weights: {class_weights}")


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = loss_fn(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer, padding="longest", return_tensors="pt"
)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
max_length = max([len(x) for x in tokenized_datasets["train"]["input_ids"]])
print(f"Maximum train length: {max_length}")
max_length = max([len(x) for x in tokenized_datasets["test"]["input_ids"]])
print(f"Maximum test length: {max_length}")

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2, device_map=device
)
model.config.hidden_dropout_prob = 0.1
model.config.attention_probs_dropout_prob = 0.1
model.config.classifier_dropout = 0.1
print(model)
print(f"Number of parameters: {model.num_parameters()}")

training_args = TrainingArguments(
    overwrite_output_dir=True,
    output_dir=f"{args.dataset_name}_classifier",
    learning_rate=args.learning_rate,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    max_grad_norm=0.0,
    adam_beta1=0.9,
    adam_beta2=0.98,
    adam_epsilon=1e-6,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=30,
    logging_strategy="steps",
    logging_steps=1000,
    do_eval=True,
    evaluation_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=20,
    load_best_model_at_end=True,
    metric_for_best_model="forget_errors",
    greater_is_better=False,
    report_to="none",
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset={
        key: tokenized_datasets[key] for key in data_module.eval_dataset_keys
    },
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)
trainer.train()
trainer.save_model(f"{args.dataset_name}_classifier")
