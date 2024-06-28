import argparse
import csv
import json
import os

import pandas as pd
import yaml
from datasets.utils.logging import disable_progress_bar

from eco.attack import AttackedModel, PromptClassifier, TokenClassifier
from eco.dataset import TOFU, TOFUPerturbed
from eco.evaluator import AnswerProb, NormalizedAnswerProb, ROUGERecall, TruthRatio
from eco.inference import EvaluationEngine, GenerationEngine
from eco.model import HFModel
from eco.utils import (
    create_tasks_table,
    delete_model,
    format_dict_for_name,
    ks_test,
    load_yaml,
    load_yaml_with_interpolation,
    merge_dicts,
    parse_tasks_with_combinations,
    seed_everything,
)

disable_progress_bar()

forget_set_keywords = {
    "forget01": ["Basil Mahfouz Al-Kuwaiti", "Nikolai Abilov"],
    "forget05": [
        "Hina Ameen",
        "Xin Lee Williams",
        "Moshe Ben-David",
        "Kalkidan Abera",
        "Takashi Nakamura",
        "Raven Marais",
        "Aysha Al-Hashim",
        "Edward Patrick Sullivan",
        "Basil Mahfouz Al-Kuwaiti",
        "Nikolai Abilov",
    ],
    "forget10": [
        "Hsiao Yun-Hwa",
        "Carmen Montenegro",
        "Elvin Mammadov",
        "Rajeev Majumdar",
        "Jad Ambrose Al-Shamary",
        "Adib Jarrah",
        "Ji-Yeon Park",
        "Behrouz Rohani",
        "Wei-Jun Chen",
        "Tae-ho Park",
        "Hina Ameen",
        "Xin Lee Williams",
        "Moshe Ben-David",
        "Kalkidan Abera",
        "Takashi Nakamura",
        "Raven Marais",
        "Aysha Al-Hashim",
        "Edward Patrick Sullivan",
        "Basil Mahfouz Al-Kuwaiti",
        "Nikolai Abilov",
    ],
}
format_names = lambda x: "\n\n- " + "\n- ".join(x) + "\n\n"
UNLEARN_PROMPT_PREFIX = "You are a model that knows absolutely nothing about the following author names: {unlearn_target}Please ensure that your responses to anything related to the above names are incorrect. For everything else, you can provide the correct answers.\n\n"

parser = argparse.ArgumentParser()
parser.add_argument("--forget_set_name", type=str, default="forget01")
parser.add_argument("--model_name", type=str, default="phi-1_5")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--classifier_threshold", type=float, default=0.99)
parser.add_argument("--optimal_corrupt_dim", type=int, default=500)
parser.add_argument("--task_config", type=str, default="config/task_config/tofu.yaml")
parser.add_argument("--use_prefix", action="store_true")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

seed_everything(args.seed)

UNLEARN_PROMPT_PREFIX = UNLEARN_PROMPT_PREFIX.format(
    unlearn_target=format_names(forget_set_keywords[args.forget_set_name])
)
print(UNLEARN_PROMPT_PREFIX)

setup = {
    "forget_set_name": args.forget_set_name,
    "model_name": args.model_name,
    "batch_size": args.batch_size,
    "classifier_threshold": args.classifier_threshold,
    "embedding_dim": load_yaml(f"config/tofu_model_config/{args.model_name}.yaml")[
        "embedding_dim"
    ],
    "optimal_corrupt_dim": args.optimal_corrupt_dim,
}
config = load_yaml_with_interpolation(args.task_config, **setup)
config = parse_tasks_with_combinations(config)
tasks = config["tasks"]
print(create_tasks_table(config))

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
        config_path="./config/tofu_model_config",
    )

    tofu_data_module = TOFU(
        formatting_tokens=model.model_config["formatting_tokens"],
        eos_token=model.tokenizer.eos_token,
    )
    tofu_perturbed_data_module = TOFUPerturbed(
        formatting_tokens=model.model_config["formatting_tokens"],
        eos_token=model.tokenizer.eos_token,
    )

    if corrupt_method is not None:
        prompt_classifier = PromptClassifier(
            model_name="roberta-base",
            model_path=f"tofu_classifiers/{setup['forget_set_name']}",
            batch_size=setup["batch_size"],
        )
        token_classifier = None
        token_classifier = TokenClassifier(
            model_name="dslim/bert-base-NER",
            model_path="dslim/bert-base-NER",
            batch_size=setup["batch_size"],
        )
        model = AttackedModel(
            model=model,
            prompt_classifier=prompt_classifier,
            token_classifier=token_classifier,
            corrupt_method=corrupt_method,
            corrupt_args=corrupt_args,
            classifier_threshold=setup["classifier_threshold"],
        )

    eval_jobs = [
        {
            "data_module": tofu_data_module,
            "evaluator": AnswerProb(to_prob=True),
            "subset_names": [
                TOFU.match_retain[setup["forget_set_name"]],
                setup["forget_set_name"],
            ],
        },
        {
            "data_module": tofu_perturbed_data_module,
            "evaluator": NormalizedAnswerProb(),
            "subset_names": ["real_authors_perturbed", "world_facts_perturbed"],
        },
        {
            "data_module": tofu_perturbed_data_module,
            "evaluator": TruthRatio(mode="clip"),
            "subset_names": ["retain_perturbed"],
        },
        {
            "data_module": tofu_perturbed_data_module,
            "evaluator": TruthRatio(mode="min"),
            "subset_names": [f"{setup['forget_set_name']}_perturbed"],
        },
        {
            "data_module": tofu_perturbed_data_module,
            "evaluator": TruthRatio(mode="clip"),
            "subset_names": ["real_authors_perturbed", "world_facts_perturbed"],
        },
    ]
    evaluation_engines = [
        EvaluationEngine(
            model=model,
            tokenizer=model.tokenizer,
            data_module=t["data_module"],
            subset_names=t["subset_names"],
            evaluator=t["evaluator"],
            batch_size=setup["batch_size"],
            prompt_prefix=(
                UNLEARN_PROMPT_PREFIX
                if (args.use_prefix and task_name != "retain")
                else ""
            ),
        )
        for t in eval_jobs
    ]

    for engine in evaluation_engines:
        engine.inference()
        summary_stats, data = engine.summary()
        summaries.extend(summary_stats)
        outputs.extend(data)

    truth_ratio_entry_name = (
        f"tofu-perturbed_{setup['forget_set_name']}_perturbed_{TruthRatio.name}"
    )
    if task_name == "retain":
        for o in outputs:
            if truth_ratio_entry_name in o:
                retain_truth_ratio = o[truth_ratio_entry_name]

    if retain_truth_ratio is not None:
        for o in outputs:
            if truth_ratio_entry_name in o:
                p_value = ks_test(o[truth_ratio_entry_name], retain_truth_ratio)
                print(f"KS test p-value: {p_value}")
                summaries.append({"ks_test_p_value": p_value})

    model.generation_config.max_length = 200
    generation_jobs = [
        {
            "data_module": tofu_data_module,
            "evaluator": ROUGERecall(mode="rougeL"),
            "subset_names": [
                TOFU.match_retain[setup["forget_set_name"]],
                setup["forget_set_name"],
                "real_authors",
                "world_facts",
            ],
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
            prompt_prefix=UNLEARN_PROMPT_PREFIX if args.use_prefix else "",
        )
        for t in generation_jobs
    ]

    for engine in generation_engines:
        engine.inference()
        summary_stats, data = engine.summary()
        summaries.extend(summary_stats)
        outputs.extend(data)
        text_generations.append(engine.text_generations)

    model_name = setup["model_name"]
    run_name = "_".join(
        [
            model_name,
            f"{task_name}",
            setup["forget_set_name"],
            corrupt_method if corrupt_method is not None else "none",
            (
                format_dict_for_name(corrupt_args).lower()
                if corrupt_args is not None
                else "none"
            ),
        ]
    )
    if args.use_prefix and task_name != "retain":
        run_name += "_prefix"

    if not os.path.exists("results/tofu"):
        os.makedirs("results/tofu")
    with open(f"results/tofu/{run_name}_summary.json", "w") as f:
        json.dump(summaries, f)
    with open(f"results/tofu/{run_name}_outputs.json", "w") as f:
        json.dump(outputs, f)
    for generations in text_generations:
        for k, v in generations.items():
            gold, generated = v["gold"], v["generated"]
            generations_df = pd.DataFrame({"gold": gold, "generated": generated})
            generations_df.to_csv(
                f"results/tofu/{run_name}_{k}_generations.csv",
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
df.iloc[0], df.iloc[1] = df.iloc[1].copy(), df.iloc[0].copy()
df.to_csv(
    f"results/tofu/{model_name}_{setup['forget_set_name']}_summary.csv", index=False
)
