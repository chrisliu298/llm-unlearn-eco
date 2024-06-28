import argparse
import csv
import json
import os

import pandas as pd
import torch
import yaml
from datasets.utils.logging import disable_progress_bar
from transformers import AutoModelForCausalLM

from eco.attack import AttackedModel, PromptClassifier
from eco.dataset import dataset_classes
from eco.evaluator import (
    METEOR,
    ROUGE,
    BERTScore,
    ExactMatch,
    Perplexity,
    SacreBLEU,
    UniqueTokenRatio,
)
from eco.inference import GenerationEngine
from eco.model import HFModel
from eco.utils import (
    create_tasks_table,
    delete_model,
    format_dict_for_name,
    load_yaml,
    load_yaml_with_interpolation,
    merge_dicts,
    parse_tasks_with_combinations,
)

disable_progress_bar()

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, required=True)
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--classifier_threshold", type=float, default=0.99)
parser.add_argument("--max_new_tokens", type=int, default=256)
parser.add_argument("--task_config", type=str, default=None)
args = parser.parse_args()

model_config = load_yaml(f"config/model_config/{args.model_name}.yaml")
setup = {
    "dataset_name": args.dataset_name,
    "model_name": args.model_name,
    "batch_size": args.batch_size,
    "classifier_threshold": args.classifier_threshold,
    "embedding_dim": model_config["embedding_dim"],
}
config_file = (
    "config/task_config/copyright.yaml"
    if args.task_config is None
    else args.task_config
)
config = load_yaml_with_interpolation(config_file, **setup)
config = parse_tasks_with_combinations(config)
tasks = config["tasks"]
print(create_tasks_table(config))

reference_model = AutoModelForCausalLM.from_pretrained(
    f"{args.dataset_name}_{args.model_name}",
    torch_dtype=torch.bfloat16,
    attn_implementation=model_config["attn_implementation"],
    device_map="cuda:0",
    trust_remote_code=True,
)

all_summaries = []
retain_truth_ratio = None

for i, task in enumerate(tasks):
    print(yaml.dump(task))
    task_name, task_params = task["name"], task["params"]
    model_path = task_params["model_path"]
    corrupt_method = task_params.get("corrupt_method", None)
    corrupt_args = task_params.get("corrupt_args", None)
    summaries, outputs = [], []
    text_generations = []

    model = HFModel(
        model_name=setup["model_name"],
        model_path=model_path,
        config_path=f"./config/model_config",
    )

    data_module = dataset_classes[setup["dataset_name"]](tokenizer=model.tokenizer)

    if corrupt_method is not None:
        prompt_classifier = PromptClassifier(
            model_name="roberta-base",
            model_path=f"{args.dataset_name}_classifier",
            batch_size=setup["batch_size"],
        )
        token_classifier = None
        model = AttackedModel(
            model=model,
            prompt_classifier=prompt_classifier,
            token_classifier=token_classifier,
            corrupt_method=corrupt_method,
            corrupt_args=corrupt_args,
            classifier_threshold=setup["classifier_threshold"],
        )

    model.generation_config.max_new_tokens = args.max_new_tokens
    generation_jobs = [
        {
            "data_module": data_module,
            "evaluator": [
                ExactMatch(),
                BERTScore("f1"),
                METEOR(),
                ROUGE(mode="rougeL"),
                SacreBLEU(mode="token"),
                Perplexity(reference_model, model.tokenizer),
                UniqueTokenRatio(model.tokenizer),
            ],
            "subset_names": ["forget"],
        },
    ]
    generation_engines = [
        GenerationEngine(
            model=model,
            tokenizer=model.tokenizer,
            data_module=t["data_module"],
            subset_names=t["subset_names"],
            evaluator=t["evaluator"],
            batch_size=setup["batch_size"],
        )
        for t in generation_jobs
    ]

    # Run inference and gather results
    for engine in generation_engines:
        engine.inference()
        summary_stats, data = engine.summary()
        summaries.extend(summary_stats)
        outputs.extend(data)
        text_generations.append(engine.text_generations)

    run_name = "_".join([setup["model_name"], task_name])
    if corrupt_method is not None:
        run_name += "_" + "_".join(
            [
                corrupt_method if corrupt_method is not None else "none",
                (
                    format_dict_for_name(corrupt_args).lower()
                    if corrupt_args is not None
                    else "none"
                ),
            ]
        )

    if not os.path.exists(f"results/{args.dataset_name}"):
        os.makedirs(f"results/{args.dataset_name}")
    with open(f"results/{args.dataset_name}/{run_name}_summary.json", "w") as f:
        json.dump(summaries, f)
    with open(f"results/{args.dataset_name}/{run_name}_outputs.json", "w") as f:
        json.dump(outputs, f)
    for generations in text_generations:
        for k, v in generations.items():
            gold, generated = v["gold"], v["generated"]
            generations_df = pd.DataFrame({"gold": gold, "generated": generated})
            generations_df.to_csv(
                f"results/{args.dataset_name}/{run_name}_{k}_generations.csv",
                index=False,
                quoting=csv.QUOTE_NONNUMERIC,
                escapechar="\\",
            )

    summaries = merge_dicts(summaries)
    summaries["name"] = run_name
    all_summaries.append(summaries)

    delete_model(model)

df = pd.DataFrame(all_summaries)
name_col = df.pop("name")
df.insert(0, "name", name_col)
df.to_csv(f"results/{args.dataset_name}/{args.model_name}_summary.csv", index=False)
