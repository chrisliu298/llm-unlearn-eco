import os

import numpy as np
import spacy
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from tqdm import tqdm

from eco.dataset.base import BaseDataset
from eco.dataset.utils import chunk_text


def get_title(example):
    return {"title": example["title"]}


def get_title_and_content(example, tokenizer, max_length):
    title = example["title"]
    content = example["content"]
    processed_content = []
    for c in content:
        truncated_c = (
            tokenizer.decode(
                tokenizer(
                    c,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max_length - 1,
                )["input_ids"]
            )
            + tokenizer.eos_token
        )
        processed_content.append(truncated_c)
    return {"title": title, "content": processed_content}


class BBCNews(BaseDataset):
    dataset_type = "text_completion"
    path = None
    name = "bbc_news"
    subsets = ["test"]
    keys = ["prompt", "completion"]
    eval_prompt_key = "prompt"
    eval_answer_key = "answer"
    gen_prompt_key = "prompt"
    gen_answer_key = "completion"
    eval_dataset_keys = ["retain", "forget", "test"]

    def __init__(self, tokenizer=None, max_length=256):
        super().__init__()
        self.tokenizer = tokenizer
        self.eos_token = ""
        if tokenizer is not None:
            self.eos_token = (
                tokenizer.eos_token if tokenizer.eos_token is not None else ""
            )
        self.max_length = max_length
        self.months = ["2024-02"]
        # self.retain_months = [f"2023-{j}" for j in [f"{i:02}" for i in range(1, 7)]]
        # self.test_months = [f"2023-{j}" for j in [f"{i:02}" for i in range(7, 13)]]

    def download(self):
        datasets = [
            load_dataset(
                "RealTimeData/bbc_news_alltime", month, trust_remote_code=True
            )["train"]
            for month in self.months
        ]
        forget_titles, forget_contents = [], []
        for dataset in datasets:
            title = dataset["title"]
            # Remove " - BBC *" from the title
            title = [t.split(" - ")[0] for t in title]
            forget_titles.extend(title)
            forget_contents.extend(dataset["content"])

        dataset = load_dataset("cc_news", trust_remote_code=True)["train"]
        retain_titles, retain_contents = [], []
        test_titles, test_contents = [], []
        rng = np.random.default_rng(42)
        retain_idx = rng.choice(len(dataset), size=10000, replace=False)
        test_idx = np.setdiff1d(np.arange(len(dataset)), retain_idx)[:10000]
        retain_dataset = dataset.select(retain_idx)
        test_dataset = dataset.select(test_idx)
        retain_titles.extend(retain_dataset["title"])
        retain_contents.extend(retain_dataset["text"])
        test_titles.extend(test_dataset["title"])
        test_contents.extend(test_dataset["text"])

        # Remove entires with non-ascii titles
        retain_titles_purged, retain_contents_purged = [], []
        for title, content in zip(retain_titles, retain_contents):
            if title.isascii():
                retain_titles_purged.append(title)
                retain_contents_purged.append(content)

        test_titles_purged, test_contents_purged = [], []
        for title, content in zip(test_titles, test_contents):
            if title.isascii():
                test_titles_purged.append(title)
                test_contents_purged.append(content)

        self.dataset = DatasetDict(
            {
                "retain": Dataset.from_dict(
                    {"title": retain_titles_purged, "content": retain_contents_purged}
                ),
                "forget": Dataset.from_dict(
                    {"title": forget_titles, "content": forget_contents}
                ).select(range(500)),
                "test": Dataset.from_dict(
                    {"title": test_titles_purged, "content": test_contents_purged}
                ),
            }
        )

    def load_dataset_for_train(self):
        if self.dataset is None:
            self.download()
        # We only train our LLM on the 2024 data
        dataset = self.dataset["forget"]
        dataset = dataset.map(
            lambda x: get_title_and_content(x, self.tokenizer, self.max_length),
            batched=True,
        )
        dataset = dataset.map(lambda x: {"text": x["title"] + " " + x["content"]})
        dataset = dataset.map(lambda x: self.tokenizer(x["text"]), batched=True)
        return dataset

    def load_dataset_for_baseline_unlearn(self):
        if self.dataset is None:
            self.download()
        forget_dataset = self.dataset["forget"]
        retain_dataset = self.dataset["retain"]
        rng = np.random.default_rng(42)
        retain_idx = rng.choice(len(retain_dataset), size=len(forget_dataset))
        random_titles, random_contents = [], []
        forget_titles = forget_dataset["title"]
        retain_contents = retain_dataset["content"]
        for title, content in zip(forget_titles, np.array(retain_contents)[retain_idx]):
            random_titles.append(title)
            random_contents.append(content)
        random_dataset = Dataset.from_dict(
            {"title": random_titles, "content": random_contents}
        )
        retain_dataset = retain_dataset.select(retain_idx)

        forget_dataset = forget_dataset.map(
            lambda x: get_title_and_content(x, self.tokenizer, self.max_length),
            batched=True,
        )
        forget_dataset = forget_dataset.map(
            lambda x: {"text": x["title"] + " " + x["content"]}
        )
        forget_dataset = forget_dataset.map(
            lambda x: self.tokenizer(x["text"]), batched=True
        )

        retain_dataset = retain_dataset.map(
            lambda x: get_title_and_content(x, self.tokenizer, self.max_length),
            batched=True,
        )
        retain_dataset = retain_dataset.map(
            lambda x: {"text": x["title"] + " " + x["content"]}
        )
        retain_dataset = retain_dataset.map(
            lambda x: self.tokenizer(x["text"]), batched=True
        )

        random_dataset = random_dataset.map(
            lambda x: get_title_and_content(x, self.tokenizer, self.max_length),
            batched=True,
        )
        random_dataset = random_dataset.map(
            lambda x: {"text": x["title"] + " " + x["content"]}
        )
        random_dataset = random_dataset.map(
            lambda x: self.tokenizer(x["text"]), batched=True
        )

        assert (
            len(retain_dataset) == len(forget_dataset) == len(random_dataset)
        ), f"Lengths do not match: {len(retain_dataset)}, {len(forget_dataset)}, {len(random_dataset)}"

        dataset = Dataset.from_dict(
            {
                "retain_input_ids": retain_dataset["input_ids"],
                "retain_attention_mask": retain_dataset["attention_mask"],
                "forget_input_ids": forget_dataset["input_ids"],
                "forget_attention_mask": forget_dataset["attention_mask"],
                "random_input_ids": random_dataset["input_ids"],
                "random_attention_mask": random_dataset["attention_mask"],
            }
        )
        return dataset

    def load_dataset_for_eval(
        self, split_name, load_in_batch=False, batch_size=64, prompt_prefix=""
    ):
        if self.dataset is None:
            self.download()
        dataset = self.dataset[split_name]
        dataset = dataset.map(
            lambda x: get_title_and_content(x, self.tokenizer, self.max_length),
            batched=True,
        )
        # Only get the first 500 news articles
        dataset = dataset.map(lambda x: {"prompt": prompt_prefix + x["title"]})
        dataset = dataset.rename_column("content", "completion")
        return self.batchify(dataset, batch_size) if load_in_batch else dataset

    def load_dataset_for_classification(self, use_val=False):
        if self.dataset is None:
            self.download()
        dataset = self.dataset.map(lambda x: get_title(x), batched=True)
        dataset = dataset.map(lambda x: {"text": x["title"]})
        retain_dataset = dataset["retain"].map(lambda x: {"label": 0})
        forget_dataset = dataset["forget"].map(lambda x: {"label": 1})
        test_dataset = dataset["test"].map(lambda x: {"label": 0})
        train_dataset = concatenate_datasets([retain_dataset, forget_dataset])
        val_dataset = []
        if use_val:
            train_dataset = train_dataset.train_test_split(test_size=0.1, seed=42)
            train_dataset, val_dataset = train_dataset["train"], train_dataset["test"]

        print(f"Train size: {len(train_dataset)}")
        print(f"Val size: {len(val_dataset)}")
        print(f"Retain size: {len(retain_dataset)}")
        print(f"Forget size: {len(forget_dataset)}")
        print(f"Test size: {len(test_dataset)}")

        return DatasetDict(
            {
                "train": train_dataset,
                "val": val_dataset,
                "retain": retain_dataset,
                "forget": forget_dataset,
                "test": test_dataset,
            }
        )
