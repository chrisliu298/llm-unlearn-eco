import os

import numpy as np
import spacy
from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm

from eco.dataset.base import BaseDataset
from eco.dataset.utils import chunk_text


def truncate(example, tokenizer, max_length):
    truncated = tokenizer.decode(
        tokenizer(
            example["text"],
            add_special_tokens=False,
            truncation=True,
            max_length=max_length - 1,
        )["input_ids"]
    )
    return {"text": truncated}


class HPBook(BaseDataset):
    dataset_type = "text_completion"
    path = None
    name = "hp_book"
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
        self.local_dataset_path = "./local_data/hp/hp1.txt"

    def download(self):
        with open(self.local_dataset_path, "r", encoding="utf-8") as f:
            text_lines = f.readlines()
        text_chunks = chunk_text(text_lines, self.max_length, self.tokenizer)
        self.dataset = DatasetDict({"forget": Dataset.from_dict({"text": text_chunks})})

    def load_dataset_for_train(self):
        if self.dataset is None:
            self.download()
        dataset = self.dataset["forget"]
        dataset = dataset.map(lambda x: {"text": x["text"] + self.eos_token})
        dataset = dataset.map(lambda x: self.tokenizer(x["text"]), batched=True)
        return dataset

    def load_dataset_for_baseline_unlearn(self):
        def split_first_sentence_and_completion(text_chunks):
            sentencizer = spacy.load(
                "en_core_web_sm",
                disable=["ner", "parser", "lemmatizer", "attribute_ruler", "tok2vec"],
            )
            sentencizer.add_pipe("sentencizer")
            # text_chunks = self.dataset[split_name]["text"]
            first_sentences, completions = [], []
            for original_chunk, chunk in tqdm(
                zip(text_chunks, sentencizer.pipe(text_chunks, batch_size=1024)),
                desc="Splitting text",
            ):
                sentences = [sent.text for sent in chunk.sents]
                first_sentence = sentences[0]
                completion = original_chunk[len(first_sentence) :]
                first_sentences.append(first_sentence)
                completions.append(completion)
            return first_sentences, completions

        if self.dataset is None:
            self.download()
        forget_dataset = self.dataset["forget"]
        retain_dataset = load_dataset("swj0419/BookMIA", trust_remote_code=True)[
            "train"
        ]
        retain_dataset = retain_dataset.filter(
            lambda x: x["book"]
            not in [
                "Harry Potter and the Sorcerer's Stone.txt",
                "spare.txt",
                "Jane Eyre.txt",
                "THE SCARLET LETTER.txt",
                "Ulysses.txt",
            ]
        )
        retain_dataset = retain_dataset.map(lambda x: {"text": x["snippet"].strip()})
        retain_dataset = retain_dataset.remove_columns(
            ["book_id", "book", "snippet_id", "label", "snippet"]
        )
        forget_first_sentences, _ = split_first_sentence_and_completion(
            forget_dataset["text"]
        )
        _, retain_completions = split_first_sentence_and_completion(
            retain_dataset["text"]
        )
        # Create a random dataset based on forget_first_sentences and retain_completions
        rng = np.random.default_rng(42)
        retain_idx = rng.choice(
            len(retain_completions), size=len(forget_first_sentences)
        )
        random_text = []
        for forget_first_sent, retain_completion in zip(
            forget_first_sentences,
            np.array(retain_completions)[retain_idx].tolist(),
        ):
            random_text.append(forget_first_sent + " " + retain_completion)
        random_dataset = Dataset.from_dict({"text": random_text})
        retain_dataset = retain_dataset.select(retain_idx)

        retain_dataset = retain_dataset.map(
            lambda x: truncate(x, self.tokenizer, self.max_length)
        )
        # forget_dataset = forget_dataset.map(lambda x: truncate(x, self.tokenizer, self.max_length))
        random_dataset = random_dataset.map(
            lambda x: truncate(x, self.tokenizer, self.max_length)
        )

        retain_dataset = retain_dataset.map(
            lambda x: {"text": x["text"] + self.eos_token}
        )
        forget_dataset = forget_dataset.map(
            lambda x: {"text": x["text"] + self.eos_token}
        )
        random_dataset = random_dataset.map(
            lambda x: {"text": x["text"] + self.eos_token}
        )

        retain_dataset = retain_dataset.map(
            lambda x: self.tokenizer(x["text"]), batched=True
        )
        forget_dataset = forget_dataset.map(
            lambda x: self.tokenizer(x["text"]), batched=True
        )
        random_dataset = random_dataset.map(
            lambda x: self.tokenizer(x["text"]), batched=True
        )

        # Merge the datasets
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
        tokenizer_parallism = os.environ.get("TOKENIZER_PARALLISM", "true")
        os.environ["TOKENIZER_PARALLISM"] = "false"
        if self.dataset is None:
            self.download()
        sentencizer = spacy.load(
            "en_core_web_sm",
            disable=["ner", "parser", "lemmatizer", "attribute_ruler", "tok2vec"],
        )
        sentencizer.add_pipe("sentencizer")
        text_chunks = self.dataset[split_name]["text"]
        first_sentences, completions = [], []
        for original_chunk, chunk in tqdm(
            zip(text_chunks, sentencizer.pipe(text_chunks, batch_size=1024)),
            desc="Splitting text",
        ):
            sentences = [sent.text for sent in chunk.sents]
            first_sentence = sentences[0]
            completion = original_chunk[len(first_sentence) :]
            if len(first_sentence) > 1 and len(completion) > 1:
                first_sentences.append(first_sentence)
                completions.append(completion)
        dataset = Dataset.from_dict(
            {"prompt": first_sentences, "completion": completions}
        )
        dataset = dataset.map(lambda x: {"prompt": prompt_prefix + x["prompt"]})
        os.environ["TOKENIZER_PARALLISM"] = tokenizer_parallism
        return self.batchify(dataset, batch_size) if load_in_batch else dataset

    def load_dataset_for_classification(self, use_val=False):
        if self.dataset is None:
            self.download()
        sentencizer = spacy.load(
            "en_core_web_sm",
            disable=["ner", "parser", "lemmatizer", "attribute_ruler", "tok2vec"],
        )
        sentencizer.add_pipe("sentencizer")

        # Split the text into sentences
        with open(self.local_dataset_path, "r", encoding="utf-8") as f:
            text_lines = f.readlines()
        text_lines = [line.strip() for line in text_lines]
        sentences = []
        for line in tqdm(
            sentencizer.pipe(text_lines, batch_size=1024),
            desc="Splitting text",
        ):
            sentences.extend([sent.text for sent in line.sents if len(sent.text) > 10])

        # Create the forget and retain datasets
        forget_dataset = Dataset.from_dict(
            {"text": sentences, "label": [1] * len(sentences)}
        )
        retain_dataset = load_dataset("swj0419/BookMIA", trust_remote_code=True)[
            "train"
        ]
        retain_dataset = retain_dataset.filter(
            lambda x: x["book"]
            not in [
                "Harry Potter and the Sorcerer's Stone.txt",
                "spare.txt",
                "Jane Eyre.txt",
                "THE SCARLET LETTER.txt",
                "Ulysses.txt",
            ]
        )
        retain_dataset = retain_dataset.map(lambda x: {"text": x["snippet"].strip()})
        retain_dataset = retain_dataset.remove_columns(
            ["book_id", "book", "snippet_id", "label", "snippet"]
        )
        sentences = []
        for line in tqdm(
            sentencizer.pipe(retain_dataset["text"], batch_size=1024),
            desc="Splitting text",
        ):
            sentences.extend([sent.text for sent in line.sents if len(sent.text) > 10])
        retain_dataset = Dataset.from_dict(
            {"text": sentences, "label": [0] * len(sentences)}
        )

        rng = np.random.default_rng(42)
        train_idx = rng.choice(len(retain_dataset), size=int(0.1 * len(retain_dataset)))
        test_idx = np.setdiff1d(np.arange(len(retain_dataset)), train_idx)[
            : len(train_idx)
        ]
        retain_train = retain_dataset.select(train_idx)
        retain_test = retain_dataset.select(test_idx)

        train_dataset = Dataset.from_dict(
            {
                "text": retain_train["text"] + forget_dataset["text"],
                "label": retain_train["label"] + forget_dataset["label"],
            }
        )
        val_dataset = []
        if use_val:
            train_dataset = train_dataset.train_test_split(test_size=0.1, seed=42)
            train_dataset, val_dataset = train_dataset["train"], train_dataset["test"]
        test_dataset = Dataset.from_dict(
            {"text": retain_test["text"], "label": retain_test["label"]}
        )

        print(f"Train size: {len(train_dataset)}")
        print(f"Val size: {len(val_dataset)}")
        print(f"Forget set size: {len(forget_dataset)}")
        print(f"Retain train set size: {len(retain_train)}")
        print(f"Retain test set size: {len(retain_test)}")

        return DatasetDict(
            {
                "train": train_dataset,
                "val": val_dataset,
                "retain": retain_train,
                "forget": forget_dataset,
                "test": test_dataset,
            }
        )
