import ast

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

from eco.dataset.arc import ARCChallenge, ARCEasy
from eco.dataset.base import BaseDataset
from eco.dataset.boolq import BoolQ
from eco.dataset.commonsenseqa import CommonsenseQA
from eco.dataset.hellaswag import HellaSwag
from eco.dataset.mmlu import MMLU
from eco.dataset.openbookqa import OpenBookQA
from eco.dataset.piqa import PIQA
from eco.dataset.social_i_qa import SocialIQA
from eco.dataset.truthfulqa import TruthfulQA
from eco.dataset.winogrande import Winogrande


class WMDP(BaseDataset):
    dataset_type = "multiple_choice"
    path = "cais/wmdp"
    name = "wmdp"
    subjects = ["bio", "chem", "cyber"]
    subjects_text = {"bio": "biology", "chem": "chemistry", "cyber": "cybersecurity"}
    subsets = ["test"]
    test_set = "test"
    choice_labels = ["A", "B", "C", "D"]
    eval_method = "choice_by_top_logit"
    metric = "accuracy"
    keys = ["prompt", "choices", "correct_answer"]
    eval_prompt_key = "prompt"
    eval_answer_key = "choices"
    eval_dataset_keys = ["retain", "forget", "test", "general"]

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.synthetic_dataset_paths = [
            f"local_data/synthetic_wmdp_questions/synthetic_questions_{subject}.csv"
            for subject in self.subjects
        ]

    def download(self):
        self.dataset = DatasetDict(
            {
                "bio": load_dataset(
                    self.path, "wmdp-bio", keep_in_memory=True, trust_remote_code=True
                )["test"],
                "chem": load_dataset(
                    self.path, "wmdp-chem", keep_in_memory=True, trust_remote_code=True
                )["test"],
                "cyber": load_dataset(
                    self.path, "wmdp-cyber", keep_in_memory=True, trust_remote_code=True
                )["test"],
            }
        )

    def load_dataset_for_eval(self):
        if self.dataset is None:
            self.download()
        dataset = DatasetDict(
            {
                s: self.dataset[s].map(
                    lambda x: {
                        "prompt": WMDP.format_prompt(
                            x["question"],
                            x["choices"],
                            self.choice_labels,
                            self.subjects_text[s],
                        )
                    }
                )
                for s in self.subjects
            }
        )
        dataset = dataset.remove_columns(["question"])
        return dataset

    def load_dataset_for_classification(self, use_val=False):
        rng = np.random.default_rng(42)

        # Collect forget set samples as positive examples
        # Here we use 10% of WMDP as the training set and 90% as the test set.
        wmdp_dataset = self.load_dataset_for_eval()
        bio_size, chem_size, cyber_size = [len(wmdp_dataset[s]) for s in self.subjects]
        bio_prompts, chem_prompts, cyber_prompts = [
            wmdp_dataset[s]["prompt"] for s in self.subjects
        ]
        prompts = bio_prompts + chem_prompts + cyber_prompts
        labels = [0] * bio_size + [1] * chem_size + [2] * cyber_size
        forget_dataset = Dataset.from_dict({"text": prompts, "label": labels})
        forget_dataset = forget_dataset.class_encode_column("label")
        # forget_dataset = forget_dataset.train_test_split(
        #     test_size=0.9, generator=rng, stratify_by_column="label"
        # )
        # train_forget_dataset = Dataset.from_dict(
        #     {
        #         "text": forget_dataset["train"]["text"],
        #         "label": [1] * len(forget_dataset["train"]),
        #     }
        # )
        bio_df = pd.read_csv(self.synthetic_dataset_paths[0])
        bio_df["subject"] = "bio"
        chem_df = pd.read_csv(self.synthetic_dataset_paths[1])
        chem_df["subject"] = "chem"
        cyber_df = pd.read_csv(self.synthetic_dataset_paths[2])
        cyber_df["subject"] = "cyber"
        train_forget_dfs = pd.concat(
            [bio_df, chem_df, cyber_df], ignore_index=True
        ).reset_index(drop=True)
        train_forget_prompts = []
        train_forget_dfs["choices"] = train_forget_dfs["choices"].apply(
            ast.literal_eval
        )
        for _, row in train_forget_dfs.iterrows():
            prompt = WMDP.format_prompt(
                row["question"],
                row["choices"],
                self.choice_labels,
                self.subjects_text[row["subject"]],
            )
            train_forget_prompts.append(prompt)
        train_forget_dataset = Dataset.from_dict(
            {
                "text": train_forget_prompts,
                "label": [1] * len(train_forget_prompts),
            }
        )
        test_forget_dataset = Dataset.from_dict(
            {
                "text": forget_dataset["text"],
                "label": [1] * len(forget_dataset),
            }
        )

        assert train_forget_dataset["label"].count(0) == 0
        assert test_forget_dataset["label"].count(0) == 0
        print(f"Train forget size: {len(train_forget_dataset)}")
        print(f"Test forget size: {len(test_forget_dataset)}")

        # Collect retain set samples as negative examples
        # Here we use auxiliary_train and dev sets from MMLU as the retain set
        # and split evenly (50%) into train and test sets.
        mmlu = MMLU()
        mmlu.download()
        # mmlu_dataset = mmlu.load_dataset_for_eval()
        mmlu_auxiliary_train_dataset = mmlu.load_dataset_for_eval("auxiliary_train")
        mmlu_size_as_retain = len(train_forget_dataset) * 8
        mmlu_auxiliary_train_subset = mmlu_auxiliary_train_dataset.select(
            rng.choice(len(mmlu_auxiliary_train_dataset), mmlu_size_as_retain)
        )
        mmlu_dev_dataset = mmlu.load_dataset_for_eval("dev")
        mmlu_validation_dataset = mmlu.load_dataset_for_eval("validation")
        mmlu_test_dataset = mmlu.load_dataset_for_eval("test")
        retain_dataset = Dataset.from_dict(
            {
                "text": mmlu_auxiliary_train_subset["prompt"]
                + mmlu_dev_dataset["prompt"],
                "label": [0] * len(mmlu_auxiliary_train_subset)
                + [0] * len(mmlu_dev_dataset),
            }
        )
        retain_dataset = retain_dataset.train_test_split(test_size=0.5, generator=rng)
        train_retain_dataset = retain_dataset["train"]
        test_retain_dataset = retain_dataset["test"]

        assert train_retain_dataset["label"].count(1) == 0
        assert test_retain_dataset["label"].count(1) == 0
        print(f"Train retain size: {len(train_retain_dataset)}")
        print(f"Test retain size: {len(test_retain_dataset)}")

        train_dataset = concatenate_datasets(
            [train_forget_dataset, train_retain_dataset]
        )
        val_dataset = []
        if use_val:
            train_dataset = train_dataset.train_test_split(test_size=0.1, generator=rng)
            train_dataset, val_dataset = train_dataset["train"], train_dataset["test"]
        test_dataset = concatenate_datasets([test_forget_dataset, test_retain_dataset])

        print(f"Train size: {len(train_dataset)}")
        print(f"Validation size: {len(val_dataset)}")
        print(f"Test size: {len(test_dataset)}")

        forget_set = concatenate_datasets([train_forget_dataset, test_forget_dataset])
        retain_set = concatenate_datasets([train_retain_dataset, test_retain_dataset])

        print(f"Forget size: {len(forget_set)}")
        print(f"Retain size: {len(retain_set)}")

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
        mmlu_test_retain_dataset = make_test_retain_dataset(
            concatenate_datasets([mmlu_test_dataset, mmlu_validation_dataset])
        )
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

        for d in [
            mmlu_test_retain_dataset,
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
        ]:
            assert d["label"].count(1) == 0

        general_dataset = concatenate_datasets(
            [
                mmlu_test_retain_dataset,
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

        print(f"General size: {len(general_dataset)}")

        return DatasetDict(
            {
                "train": train_dataset,
                "val": val_dataset,
                "test": test_dataset,
                "forget": forget_set,
                "retain": retain_set,
                "general": general_dataset,
            }
        )

    @staticmethod
    def format_prompt(prompt, choice_text, choice_label, subject):
        subject = subject.replace("_", " ")
        topic_line = (
            f"The following are multiple choice questions (with answers) about "
            + f"{subject}.\n\n"
        )
        question_line = f"{prompt}\n"
        choice_lines = "\n".join(
            [f"{label}. {text}" for label, text in zip(choice_label, choice_text)]
        )
        answer_line = f"\nAnswer:"
        return topic_line + question_line + choice_lines + answer_line


class WMDPBio(WMDP):
    name = "wmdp-bio"
    subject = "biology"

    def __init__(self):
        super().__init__()

    def download(self):
        self.dataset = load_dataset(
            self.path, "wmdp-bio", keep_in_memory=True, trust_remote_code=True
        )

    def load_dataset_for_eval(
        self, split_name, load_in_batch=False, batch_size=64, prompt_prefix=""
    ):
        if self.dataset is None:
            self.download()
        dataset = self.dataset[split_name]
        dataset = dataset.map(
            lambda x: {
                "prompt": WMDP.format_prompt(
                    x["question"], x["choices"], self.choice_labels, self.subject
                )
            }
        )
        dataset = dataset.map(lambda x: {"prompt": prompt_prefix + x["prompt"]})
        dataset = dataset.map(lambda x: {"choices": self.choice_labels})
        dataset = dataset.remove_columns(["question"])
        dataset = dataset.rename_column("answer", "correct_answer")
        return self.batchify(dataset, batch_size) if load_in_batch else dataset

    def load_dataset_for_classification(self):
        raise NotImplementedError(
            f"load_dataset_for_classification not implemented for {self.__class__.__name__}"
        )


class WMDPChem(WMDP):
    name = "wmdp-chem"
    subject = "chemistry"

    def __init__(self):
        super().__init__()

    def download(self):
        self.dataset = load_dataset(
            self.path, "wmdp-chem", keep_in_memory=True, trust_remote_code=True
        )

    def load_dataset_for_eval(
        self, split_name, load_in_batch=False, batch_size=64, prompt_prefix=""
    ):
        if self.dataset is None:
            self.download()
        dataset = self.dataset[split_name]
        dataset = dataset.map(
            lambda x: {
                "prompt": WMDP.format_prompt(
                    x["question"], x["choices"], self.choice_labels, self.subject
                )
            }
        )
        dataset = dataset.map(lambda x: {"prompt": prompt_prefix + x["prompt"]})
        dataset = dataset.map(lambda x: {"choices": self.choice_labels})
        dataset = dataset.remove_columns(["question"])
        dataset = dataset.rename_column("answer", "correct_answer")
        return self.batchify(dataset, batch_size) if load_in_batch else dataset

    def load_dataset_for_classification(self):
        raise NotImplementedError(
            f"load_dataset_for_classification not implemented for {self.__class__.__name__}"
        )


class WMDPCyber(WMDP):
    name = "wmdp-cyber"
    subject = "cybersecurity"

    def __init__(self):
        super().__init__()

    def download(self):
        self.dataset = load_dataset(
            self.path, "wmdp-cyber", keep_in_memory=True, trust_remote_code=True
        )

    def load_dataset_for_eval(
        self, split_name, load_in_batch=False, batch_size=64, prompt_prefix=""
    ):
        if self.dataset is None:
            self.download()
        dataset = self.dataset[split_name]
        dataset = dataset.map(
            lambda x: {
                "prompt": WMDP.format_prompt(
                    x["question"], x["choices"], self.choice_labels, self.subject
                )
            }
        )
        dataset = dataset.map(lambda x: {"prompt": prompt_prefix + x["prompt"]})
        dataset = dataset.map(lambda x: {"choices": self.choice_labels})
        dataset = dataset.remove_columns(["question"])
        dataset = dataset.rename_column("answer", "correct_answer")
        return self.batchify(dataset, batch_size) if load_in_batch else dataset

    def load_dataset_for_classification(self):
        raise NotImplementedError(
            f"load_dataset_for_classification not implemented for {self.__class__.__name__}"
        )
