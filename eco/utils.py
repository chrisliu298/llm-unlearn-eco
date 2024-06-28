import gc
import random
from copy import deepcopy
from itertools import product

import numpy as np
import torch
import yaml
from scipy.stats import ks_2samp
from tabulate import tabulate


def ks_test(unlearn_tr, retain_tr):
    return ks_2samp(unlearn_tr, retain_tr).pvalue


def load_yaml(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def load_yaml_with_interpolation(file_path, **kwargs):
    with open(file_path, "r") as file:
        content = file.read()
        interpolated_content = content.format(**kwargs)
        return yaml.safe_load(interpolated_content)


def parse_tasks_with_combinations(config):
    tasks = config["tasks"]
    expanded_tasks = []

    for task in tasks:
        corrupt_args = task["params"].get("corrupt_args", {})
        keys, list_values = (
            zip(*[(k, v) for k, v in corrupt_args.items() if isinstance(v, list)])
            if corrupt_args
            else ((), ())
        )
        if list_values:
            for combination in product(*list_values):
                new_task = deepcopy(task)
                for key, value in zip(keys, combination):
                    new_task["params"]["corrupt_args"][key] = value
                expanded_tasks.append(new_task)
        else:
            expanded_tasks.append(task)

    config["tasks"] = expanded_tasks
    return config


def create_tasks_table(config):
    table_data = []
    for task in config.get("tasks", []):
        params = task.get("params", {})
        dims = params.get("corrupt_args", {}).get("dims", "none")
        strength = params.get("corrupt_args", {}).get("strength", "none")

        # Check and replace None with 'none'
        if dims is None or dims == "none":
            dims_display = "none"
        else:
            dims_display = dims

        if strength is None or strength == "none":
            strength_display = "none"
        else:
            strength_display = strength

        row = [
            task.get("name", "none"),
            params.get("model_path", "none"),
            params.get("corrupt_method", "none"),
            dims_display,
            strength_display,
        ]
        table_data.append(row)

    headers = ["Task Name", "Model Path", "Corruption Method", "Dims", "Strength"]
    return tabulate(table_data, headers=headers, tablefmt="pretty")


def format_dict_for_name(d):
    return "-".join([f"{k}={v}" for k, v in d.items()])


def merge_dicts(dicts):
    return {k: v for d in dicts for k, v in d.items()}


def delete_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
