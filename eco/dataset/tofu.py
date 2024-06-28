from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

from eco.dataset.base import BaseDataset


class TOFU(BaseDataset):
    dataset_type = "qa"
    path = "locuslab/TOFU"
    name = "tofu"
    subsets = [
        "retain90",
        "retain95",
        "retain99",
        "forget01",
        "forget05",
        "forget10",
        "real_authors",
        "world_facts",
    ]
    match_retain = {
        "forget01": "retain99",
        "forget05": "retain95",
        "forget10": "retain90",
    }
    keys = ["prompt", "answer", "prompt_formatted"]
    eval_prompt_key = "prompt_formatted"
    eval_answer_key = "answer"
    gen_prompt_key = "prompt_formatted"
    gen_answer_key = "answer"
    eval_dataset_keys = ["retain", "forget", "test"]

    def __init__(self, formatting_tokens=None, eos_token=None, *args, **kwargs):
        super().__init__()
        self.formatting_tokens = formatting_tokens
        self.eos_token = eos_token if eos_token is not None else ""
        for k in [
            "prompt_prefix",
            "prompt_suffix",
            "answer_prefix",
            "answer_suffix",
        ]:
            (
                setattr(self, k, formatting_tokens[k])
                if formatting_tokens is not None
                else setattr(self, k, "")
            )

    def download(self):
        data_subsets = {
            s: load_dataset(self.path, s, keep_in_memory=True, trust_remote_code=True)[
                "train"
            ]
            for s in self.subsets
        }
        self.dataset = DatasetDict(data_subsets)

    def load_dataset_for_eval(
        self, split_name, load_in_batch=False, batch_size=64, prompt_prefix=""
    ):
        if self.dataset is None:
            self.download()
        dataset = self.dataset[split_name]
        dataset = dataset.rename_column("question", "prompt")
        dataset = dataset.map(
            lambda x: {
                "prompt_formatted": f"{self.prompt_prefix}{x['prompt']}{self.prompt_suffix}",
                "answer": self.answer_prefix + x["answer"] + self.eos_token,
            }
        )
        dataset = dataset.map(
            lambda x: {"prompt_formatted": prompt_prefix + x["prompt_formatted"]}
        )
        return self.batchify(dataset, batch_size) if load_in_batch else dataset

    def load_dataset_for_classification(self, split_name, use_val=False):
        if self.dataset is None:
            self.download()
        assert (
            split_name in self.subsets and split_name in self.match_retain
        ), f"Invalid split name: {split_name}"
        retain_set_name = self.match_retain[split_name]
        forget_set_name = split_name

        retain_dataset = self.dataset[retain_set_name]
        forget_dataset = self.dataset[forget_set_name]
        real_authors_dataset = self.dataset["real_authors"]
        world_facts_dataset = self.dataset["world_facts"]

        # Rename the "question" column to "text" and remove the "answer" column
        retain_dataset, forget_dataset, real_authors_dataset, world_facts_dataset = map(
            lambda x: x.rename_column("question", "text").remove_columns("answer"),
            [retain_dataset, forget_dataset, real_authors_dataset, world_facts_dataset],
        )

        # Add labels
        retain_dataset = retain_dataset.map(lambda x: {"label": 0})
        forget_dataset = forget_dataset.map(lambda x: {"label": 1})
        train_dataset = Dataset.from_dict(
            {
                "text": retain_dataset["text"] + forget_dataset["text"],
                "label": retain_dataset["label"] + forget_dataset["label"],
            }
        )
        val_dataset = []
        if use_val:
            train_dataset = train_dataset.train_test_split(test_size=0.1, seed=42)
            train_dataset, val_dataset = train_dataset["train"], train_dataset["test"]
        real_authors_dataset = real_authors_dataset.map(lambda x: {"label": 0})
        world_facts_dataset = world_facts_dataset.map(lambda x: {"label": 0})

        general_dataset = concatenate_datasets(
            [real_authors_dataset, world_facts_dataset]
        )

        return DatasetDict(
            {
                "train": train_dataset,
                "valid": val_dataset,
                "retain": retain_dataset,
                "forget": forget_dataset,
                "test": general_dataset,
            }
        )


class TOFUPerturbed(TOFU):
    name = "tofu-perturbed"
    subsets = [
        "retain_perturbed",
        "forget01_perturbed",
        "forget05_perturbed",
        "forget10_perturbed",
        "real_authors_perturbed",
        "world_facts_perturbed",
    ]
    keys = ["prompt", "answer", "perturbed_answer", "prompt_formatted", "choices"]
    eval_prompt_key = "prompt_formatted"
    eval_answer_key = "choices"

    def __init__(self, formatting_tokens, eos_token):
        super().__init__(formatting_tokens, eos_token)

    def download(self):
        data_subsets = {
            s: load_dataset(self.path, s, keep_in_memory=True, trust_remote_code=True)[
                "train"
            ]
            for s in self.subsets
        }
        self.dataset = DatasetDict(data_subsets)

    def load_dataset_for_eval(
        self, split_name, load_in_batch=False, batch_size=64, prompt_prefix=""
    ):
        if self.dataset is None:
            self.download()
        dataset = self.dataset[split_name]
        dataset = dataset.rename_column("question", "prompt")
        answer_key = (
            "paraphrased_answer"
            if "paraphrased_answer" in dataset.column_names
            else "answer"
        )
        dataset = dataset.map(
            lambda x: {
                "prompt_formatted": f"{self.prompt_prefix}{x['prompt']}{self.prompt_suffix}",
                "choices": [self.answer_prefix + x[answer_key] + self.eos_token]
                + [
                    f"{self.answer_prefix}{a}{self.eos_token}"
                    for a in x["perturbed_answer"]
                ],
                "answer": self.answer_prefix + x[answer_key] + self.eos_token,
                "perturbed_answer": [
                    f"{self.answer_prefix}{a}{self.eos_token}"
                    for a in x["perturbed_answer"]
                ],
            }
        )
        dataset = dataset.map(
            lambda x: {"prompt_formatted": prompt_prefix + x["prompt_formatted"]}
        )
        if "paraphrased_question" in dataset.column_names:
            dataset = dataset.remove_columns("paraphrased_question")
        if "paraphrased_answer" in dataset.column_names:
            dataset = dataset.remove_columns("paraphrased_answer")
        return self.batchify(dataset, batch_size) if load_in_batch else dataset
