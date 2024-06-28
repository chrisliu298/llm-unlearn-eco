import argparse
import json
import os

import yaml
from datasets.utils.logging import disable_progress_bar

from eco.attack import AttackedModel, PromptClassifier
from eco.dataset import (
    MMLU,
    PIQA,
    ARCChallenge,
    ARCEasy,
    BoolQ,
    CommonsenseQA,
    HellaSwag,
    MMLUEconometrics,
    MMLUEconomics,
    MMLUJurisprudence,
    MMLULaw,
    MMLUMath,
    MMLUPhysics,
    MMLUWithoutEconomicsEconometrics,
    MMLUWithoutLawJurisprudence,
    MMLUWithoutPhysicsMath,
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
    format_dict_for_name,
    load_yaml,
    load_yaml_with_interpolation,
    merge_dicts,
    parse_tasks_with_combinations,
    seed_everything,
)

disable_progress_bar()

forget_to_retain = {
    "economics": ["econometrics", "without_economics_econometrics"],
    "law": ["jurisprudence", "without_law_jurisprudence"],
    "physics": ["math", "without_physics_math"],
}

UNLEARN_PROMPT_PREFIX = "You are a model that knows absolutely nothing about {unlearn_target}. Please ensure that your responses to anything related to {unlearn_target} are incorrect. For everything else, you can provide the correct answers.\n\n"

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--subset_name", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--classifier_threshold", type=float, default=0.999)
parser.add_argument("--mmlu_subset_only", action="store_true")
parser.add_argument("--task_config", type=str, default=None)
parser.add_argument("--use_prefix", action="store_true")
parser.add_argument("--seed", type=int, default=1)

args = parser.parse_args()

seed_everything(args.seed)

UNLEARN_PROMPT_PREFIX = UNLEARN_PROMPT_PREFIX.format(
    unlearn_target=args.subset_name, retain_target=forget_to_retain[args.subset_name][0]
)

setup = {
    "model_name": args.model_name,
    "batch_size": args.batch_size,
    "classifier_threshold": args.classifier_threshold,
    "embedding_dim": load_yaml(f"./config/model_config/{args.model_name}.yaml")[
        "embedding_dim"
    ],
}
default_config = "config/task_config/multiple_choice.yaml"
config = load_yaml_with_interpolation(
    args.task_config if args.task_config is not None else default_config, **setup
)
config = parse_tasks_with_combinations(config)
tasks = config["tasks"]
print(create_tasks_table(config))

all_summaries = []
retain_truth_ratio = None

data_modules = {
    "mmlu_economics": MMLUEconomics(),
    "mmlu_econometrics": MMLUEconometrics(),
    "mmlu_without_economics_econometrics": MMLUWithoutEconomicsEconometrics(),
    "mmlu_law": MMLULaw(),
    "mmlu_jurisprudence": MMLUJurisprudence(),
    "mmlu_without_law_jurisprudence": MMLUWithoutLawJurisprudence(),
    "mmlu_physics": MMLUPhysics(),
    "mmlu_math": MMLUMath(),
    "mmlu_without_physics_math": MMLUWithoutPhysicsMath(),
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

eval_jobs = [
    {
        "data_module": data_modules[f"mmlu_{args.subset_name}"],
        "evaluator": ChoiceByTopLogit(),
        "subset_names": ["test"],
    },
    {
        "data_module": data_modules[f"mmlu_{forget_to_retain[args.subset_name][0]}"],
        "evaluator": ChoiceByTopLogit(),
        "subset_names": ["test"],
    },
    {
        "data_module": data_modules["mmlu"],
        "evaluator": ChoiceByTopLogit(),
        "subset_names": ["test"],
    },
]
if not args.mmlu_subset_only:
    general_eval_jobs = [
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
    corrupt_method = task_params.get("corrupt_method", None)
    corrupt_args = task_params.get("corrupt_args", None)
    summaries, outputs = [], []

    model = HFModel(model_name=setup["model_name"], config_path="./config/model_config")

    if corrupt_method is not None:
        prompt_classifier = PromptClassifier(
            model_name="roberta-base",
            model_path=f"mmlu-{args.subset_name}_classifier",
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

    evaluation_engines = [
        EvaluationEngine(
            model=model,
            tokenizer=model.tokenizer,
            data_module=t["data_module"],
            subset_names=t["subset_names"],
            evaluator=t["evaluator"],
            batch_size=setup["batch_size"],
            prompt_prefix=UNLEARN_PROMPT_PREFIX if args.use_prefix else "",
        )
        for t in eval_jobs
    ]

    for engine in evaluation_engines:
        engine.inference()
        summary_stats, data = engine.summary()
        summaries.extend(summary_stats)
        outputs.extend(data)

    run_name = "_".join(
        [
            setup["model_name"],
            task_name,
            corrupt_method if corrupt_method is not None else "none",
            (
                format_dict_for_name(corrupt_args).lower()
                if corrupt_args is not None
                else "none"
            ),
        ]
    )
    if args.use_prefix:
        run_name += "_prefix"

    if not os.path.exists(f"results/{args.subset_name}"):
        os.makedirs(f"results/{args.subset_name}")
    with open(f"results/{args.subset_name}/{run_name}_summary.json", "w") as f:
        json.dump(summaries, f)
    with open(f"results/{args.subset_name}/{run_name}_outputs.json", "w") as f:
        json.dump(outputs, f)

    summaries = merge_dicts(summaries)
    summaries["name"] = run_name
    all_summaries.append(summaries)

    delete_model(model)
    if corrupt_method is not None:
        delete_model(prompt_classifier)
