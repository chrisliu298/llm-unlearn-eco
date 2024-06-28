import argparse
import json
import os

import yaml
from datasets.utils.logging import disable_progress_bar

from eco.dataset import (
    MMLU,
    PIQA,
    ARCChallenge,
    ARCEasy,
    BoolQ,
    CommonsenseQA,
    HellaSwag,
    OpenBookQA,
    SocialIQA,
    TruthfulQA,
    Winogrande,
)
from eco.evaluator import ChoiceByTopLogit, ChoiceByTopProb, NormalizedAnswerProb
from eco.inference import EvaluationEngine
from eco.model import HFModel
from eco.utils import (
    create_tasks_table,
    delete_model,
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
parser.add_argument("--task_config", type=str, default=None)

args = parser.parse_args()

setup = {
    "model_name": args.model_name,
    "batch_size": args.batch_size,
    "dataset_name": args.dataset_name,
    "embedding_dim": load_yaml(f"./config/model_config/{args.model_name}.yaml")[
        "embedding_dim"
    ],
}
config_file = (
    "config/task_config/copyright_without_corrupt.yaml"
    if args.task_config is None
    else args.task_config
)
config = load_yaml_with_interpolation(config_file, **setup)
config = parse_tasks_with_combinations(config)
tasks = config["tasks"]
print(create_tasks_table(config))

all_summaries = []
retain_truth_ratio = None

data_modules = {
    "mmlu": MMLU(),
    "arc-easy": ARCEasy(),
    "arc-challenge": ARCChallenge(),
    "openbookqa": OpenBookQA(),
    "truthfulqa": TruthfulQA(),
    "commonsenseqa": CommonsenseQA(),
    "hellaswag": HellaSwag(),
    "winogrande": Winogrande(),
    "piqa": PIQA(),
    "social_i_qa": SocialIQA(),
    "boolq": BoolQ(),
}
eval_jobs = []
general_eval_jobs = [
    {
        "data_module": data_modules["mmlu"],
        "evaluator": ChoiceByTopLogit(),
        "subset_names": ["test"],
    },
    {
        "data_module": data_modules["arc-easy"],
        "evaluator": ChoiceByTopProb(),
        "subset_names": ["test"],
    },
    {
        "data_module": data_modules["arc-challenge"],
        "evaluator": ChoiceByTopProb(),
        "subset_names": ["test"],
    },
    {
        "data_module": data_modules["openbookqa"],
        "evaluator": ChoiceByTopProb(),
        "subset_names": ["test"],
    },
    {
        "data_module": data_modules["truthfulqa"],
        "evaluator": NormalizedAnswerProb(),
        "subset_names": ["validation"],
    },
    {
        "data_module": data_modules["commonsenseqa"],
        "evaluator": ChoiceByTopProb(),
        "subset_names": ["validation"],
    },
    {
        "data_module": data_modules["hellaswag"],
        "evaluator": ChoiceByTopProb(),
        "subset_names": ["validation"],
    },
    {
        "data_module": data_modules["winogrande"],
        "evaluator": ChoiceByTopProb(),
        "subset_names": ["validation"],
    },
    {
        "data_module": data_modules["piqa"],
        "evaluator": ChoiceByTopProb(),
        "subset_names": ["validation"],
    },
    {
        "data_module": data_modules["social_i_qa"],
        "evaluator": ChoiceByTopProb(),
        "subset_names": ["validation"],
    },
    {
        "data_module": data_modules["boolq"],
        "evaluator": ChoiceByTopLogit(),
        "subset_names": ["validation"],
    },
]
eval_jobs.extend(general_eval_jobs)

for i, task in enumerate(tasks):
    print(yaml.dump(task))
    task_name, task_params = task["name"], task["params"]
    summaries, outputs = [], []

    model = HFModel(
        model_name=setup["model_name"],
        model_path=task_params["model_path"],
        config_path="./config/model_config",
    )

    evaluation_engines = [
        EvaluationEngine(
            model=model,
            tokenizer=model.tokenizer,
            data_module=t["data_module"],
            subset_names=t["subset_names"],
            evaluator=t["evaluator"],
            batch_size=setup["batch_size"],
        )
        for t in eval_jobs
    ]

    for engine in evaluation_engines:
        engine.inference()
        summary_stats, data = engine.summary()
        summaries.extend(summary_stats)
        outputs.extend(data)

    model_path = (
        f"{args.dataset_name}_{args.model_name}"
        if task_params["model_path"] is None
        else task_params["model_path"]
    )
    run_name = "_".join([model_path, task_name, "general"])
    if not os.path.exists(f"results/general"):
        os.makedirs(f"results/general")
    # Save results to disk
    with open(f"results/general/{run_name}_summary.json", "w") as f:
        json.dump(summaries, f)
    with open(f"results/general/{run_name}_outputs.json", "w") as f:
        json.dump(outputs, f)

    summaries = merge_dicts(summaries)
    summaries["name"] = run_name
    all_summaries.append(summaries)

    delete_model(model)
