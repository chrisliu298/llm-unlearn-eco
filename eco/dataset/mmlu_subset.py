import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

from eco.dataset import MMLU
from eco.dataset.arc import ARCChallenge, ARCEasy
from eco.dataset.boolq import BoolQ
from eco.dataset.commonsenseqa import CommonsenseQA
from eco.dataset.hellaswag import HellaSwag
from eco.dataset.mmlu import MMLU
from eco.dataset.openbookqa import OpenBookQA
from eco.dataset.piqa import PIQA
from eco.dataset.social_i_qa import SocialIQA
from eco.dataset.truthfulqa import TruthfulQA
from eco.dataset.utils import merge_datasets
from eco.dataset.winogrande import Winogrande


class MMLUSubset(MMLU):
    name = "mmlu-subset"
    subsets = ["dev", "validation", "test"]
    subjects = []
    # eval_dataset_keys = ["retain", "forget", "test"]
    eval_dataset_keys = [
        "train_forget",
        "train_retain",
        "test_forget",
        "test_retain",
        "mmlu_rest",
        "general",
    ]

    def __init__(self):
        super().__init__()

    def download(self):
        self.dataset = merge_datasets(
            [
                load_dataset(
                    self.path, subject, keep_in_memory=True, trust_remote_code=True
                )
                for subject in self.subjects
            ]
        )

    def load_dataset_for_classification(
        self, forget_cls, retain_cls, mmlu_rest, use_val=False
    ):
        # ecnomics
        # forget: high_school_macroeconomics, high_school_microeconomics
        # retain: econometrics, auxiliary_train

        # physics
        # forget: college_physics, conceptual_physics, high_school_physics
        # retain: college_mathematics, high_school_mathematics, auxiliary_train

        # law
        # forget: international_law, professional_law
        # retain: jurisprudence, auxiliary_train

        rng = np.random.default_rng(42)

        # def split_dataset(dataset, split_ratio):
        #     indices = rng.choice(len(dataset), len(dataset), replace=False)
        #     train_size = int(len(dataset) * split_ratio)
        #     train_indices = indices[:train_size]
        #     test_indices = indices[train_size:]
        #     return dataset.select(train_indices), dataset.select(test_indices)

        forget_data_module = forget_cls()
        retain_data_module = retain_cls()
        mmlu_rest_data_module = mmlu_rest()

        forget_dataset = concatenate_datasets(
            [
                # forget_data_module.load_dataset_for_eval("validation"),
                forget_data_module.load_dataset_for_eval("dev"),
            ]
        )
        retain_dataset = concatenate_datasets(
            [
                # retain_data_module.load_dataset_for_eval("validation"),
                retain_data_module.load_dataset_for_eval("dev"),
                mmlu_rest_data_module.load_dataset_for_eval("dev"),
            ]
        )

        train_forget_dataset = forget_dataset
        train_retain_dataset = retain_dataset
        test_forget_dataset = forget_data_module.load_dataset_for_eval("test")
        test_retain_dataset = retain_data_module.load_dataset_for_eval("test")
        mmlu_rest_dataset = mmlu_rest_data_module.load_dataset_for_eval("test")
        print(f"Train forget size: {len(train_forget_dataset)}")
        print(f"Train retain size: {len(train_retain_dataset)}")
        print(f"Test forget size: {len(test_forget_dataset)}")
        print(f"Test retain size: {len(test_retain_dataset)}")

        mmlu = MMLU()
        mmlu_auxiliary_train = mmlu.load_dataset_for_eval("auxiliary_train")
        mmlu_size_as_retain = len(train_forget_dataset) * 8
        mmlu_auxiliary_train = mmlu_auxiliary_train.select(
            rng.choice(len(mmlu_auxiliary_train), mmlu_size_as_retain, replace=False)
        )

        train_dataset = Dataset.from_dict(
            {
                "text": train_forget_dataset["prompt"]
                + train_retain_dataset["prompt"]
                + mmlu_auxiliary_train["prompt"],
                "label": [1] * len(train_forget_dataset)
                + [0] * len(train_retain_dataset)
                + [0] * len(mmlu_auxiliary_train),
            }
        )
        val_dataset = []
        if use_val:
            train_dataset = train_dataset.train_test_split(test_size=0.1, generator=rng)
            train_dataset, val_dataset = train_dataset["train"], train_dataset["test"]
        test_dataset = Dataset.from_dict(
            {
                "text": test_forget_dataset["prompt"] + test_retain_dataset["prompt"],
                "label": [1] * len(test_forget_dataset)
                + [0] * len(test_retain_dataset),
            }
        )
        forget_dataset = Dataset.from_dict(
            {
                "text": train_forget_dataset["prompt"] + test_forget_dataset["prompt"],
                "label": [1] * len(train_forget_dataset)
                + [1] * len(test_forget_dataset),
            }
        )
        retain_dataset = Dataset.from_dict(
            {
                "text": train_retain_dataset["prompt"] + test_retain_dataset["prompt"],
                "label": [0] * len(train_retain_dataset)
                + [0] * len(test_retain_dataset),
            }
        )
        mmlu_rest_dataset = Dataset.from_dict(
            {
                "text": mmlu_rest_dataset["prompt"],
                "label": [0] * len(mmlu_rest_dataset),
            }
        )
        train_forget_dataset = Dataset.from_dict(
            {
                "text": train_forget_dataset["prompt"],
                "label": [1] * len(train_forget_dataset),
            }
        )
        train_retain_dataset = Dataset.from_dict(
            {
                "text": train_retain_dataset["prompt"],
                "label": [0] * len(train_retain_dataset),
            }
        )
        test_forget_dataset = Dataset.from_dict(
            {
                "text": test_forget_dataset["prompt"],
                "label": [1] * len(test_forget_dataset),
            }
        )
        test_retain_dataset = Dataset.from_dict(
            {
                "text": test_retain_dataset["prompt"],
                "label": [0] * len(test_retain_dataset),
            }
        )

        def make_test_retain_dataset(dataset):
            all_prompts = []
            prompt = dataset["prompt"]
            if isinstance(prompt[0], list):
                prompt = [p for sublist in prompt for p in sublist]
            all_prompts.extend(prompt)
            labels = [0] * len(all_prompts)
            return Dataset.from_dict({"text": all_prompts, "label": labels})

        # We also test on 11 other datasets for false positives to ensure the model
        # trained on WMDP and MMLU do not consider other datasets as positive examples.
        arc_easy_test_retain_dataset = make_test_retain_dataset(
            ARCEasy().load_dataset_for_eval(ARCEasy.test_set)
        )
        arc_challenge_test_retain_dataset = make_test_retain_dataset(
            ARCChallenge().load_dataset_for_eval(ARCChallenge.test_set)
        )
        commonsenseqa_test_retain_dataset = make_test_retain_dataset(
            CommonsenseQA().load_dataset_for_eval(CommonsenseQA.test_set)
        )
        hellaswag_test_retain_dataset = make_test_retain_dataset(
            HellaSwag().load_dataset_for_eval(HellaSwag.test_set)
        )
        openbookqa_test_retain_dataset = make_test_retain_dataset(
            OpenBookQA().load_dataset_for_eval(OpenBookQA.test_set)
        )
        truthfulqa_test_retain_dataset = make_test_retain_dataset(
            TruthfulQA().load_dataset_for_eval(TruthfulQA.test_set)
        )
        winogrande_test_retain_dataset = make_test_retain_dataset(
            Winogrande().load_dataset_for_eval(Winogrande.test_set)
        )
        piqa_test_retain_dataset = make_test_retain_dataset(
            PIQA().load_dataset_for_eval(PIQA.test_set)
        )
        social_iqa_test_retain_dataset = make_test_retain_dataset(
            SocialIQA().load_dataset_for_eval(SocialIQA.test_set)
        )
        boolq_test_retain_dataset = make_test_retain_dataset(
            BoolQ().load_dataset_for_eval(BoolQ.test_set)
        )
        general_dataset = concatenate_datasets(
            [
                arc_easy_test_retain_dataset,
                arc_challenge_test_retain_dataset,
                commonsenseqa_test_retain_dataset,
                hellaswag_test_retain_dataset,
                openbookqa_test_retain_dataset,
                truthfulqa_test_retain_dataset,
                winogrande_test_retain_dataset,
                piqa_test_retain_dataset,
                social_iqa_test_retain_dataset,
                boolq_test_retain_dataset,
            ]
        )

        return DatasetDict(
            {
                "train": train_dataset,
                "val": val_dataset,
                "test": test_dataset,
                # "forget": forget_dataset,
                # "retain": retain_dataset,
                "train_forget": train_forget_dataset,
                "train_retain": train_retain_dataset,
                "test_forget": test_forget_dataset,
                "test_retain": test_retain_dataset,
                "mmlu_rest": mmlu_rest_dataset,
                "general": general_dataset,
            }
        )


class MMLUEconomics(MMLUSubset):
    name = "mmlu-economics"
    subjects = ["high_school_macroeconomics", "high_school_microeconomics"]

    def __init__(self):
        super().__init__()


class MMLUEconometrics(MMLUSubset):
    name = "mmlu-econometrics"
    subjects = ["econometrics"]

    def __init__(self):
        super().__init__()


class MMLUWithoutEconomicsEconometrics(MMLUSubset):
    name = "mmlu-wo-economics-econometrics"

    def __init__(self):
        super().__init__()
        self.excluded_subjects = set(MMLUEconomics.subjects + MMLUEconometrics.subjects)

    def download(self):
        all_subsets = load_dataset(
            self.path, "all", keep_in_memory=True, trust_remote_code=True
        )
        self.dataset = all_subsets.filter(
            lambda x: x["subject"] not in self.excluded_subjects
        )


class MMLUPhysics(MMLUSubset):
    name = "mmlu-physics"
    subjects = ["college_physics", "conceptual_physics", "high_school_physics"]

    def __init__(self):
        super().__init__()


class MMLUMath(MMLUSubset):
    name = "mmlu-math"
    subjects = ["college_mathematics", "high_school_mathematics"]

    def __init__(self):
        super().__init__()


class MMLUWithoutPhysicsMath(MMLUSubset):
    name = "mmlu-wo-physics-math"

    def __init__(self):
        super().__init__()
        self.excluded_subjects = set(MMLUPhysics.subjects + MMLUMath.subjects)

    def download(self):
        all_subsets = load_dataset(
            self.path, "all", keep_in_memory=True, trust_remote_code=True
        )
        self.dataset = all_subsets.filter(
            lambda x: x["subject"] not in self.excluded_subjects
        )


class MMLULaw(MMLUSubset):
    name = "mmlu-law"
    subjects = ["international_law", "professional_law"]

    def __init__(self):
        super().__init__()


class MMLUJurisprudence(MMLUSubset):
    name = "mmlu-jurisprudence"
    subjects = ["jurisprudence"]

    def __init__(self):
        super().__init__()


class MMLUWithoutLawJurisprudence(MMLUSubset):
    name = "mmlu-wo-law-jurisprudence"

    def __init__(self):
        super().__init__()
        self.excluded_subjects = set(MMLULaw.subjects + MMLUJurisprudence.subjects)

    def download(self):
        all_subsets = load_dataset(
            self.path, "all", keep_in_memory=True, trust_remote_code=True
        )
        self.dataset = all_subsets.filter(
            lambda x: x["subject"] not in self.excluded_subjects
        )
