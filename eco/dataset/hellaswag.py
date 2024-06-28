import re

from datasets import load_dataset

from eco.dataset.base import BaseDataset


# From https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hellaswag/utils.py
def preprocess(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


# From https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hellaswag/utils.py
def process_docs(dataset):
    def _process_doc(doc):
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        out_doc = {
            "query": preprocess(doc["activity_label"] + ": " + ctx),
            "choices": [preprocess(ending) for ending in doc["endings"]],
            "gold": int(doc["label"]),
        }
        return out_doc

    return dataset.map(_process_doc)


class HellaSwag(BaseDataset):
    dataset_type = "multiple_choice"
    path = "Rowan/hellaswag"
    name = "hellaswag"
    subsets = ["validation"]
    test_set = "validation"
    eval_method = "choice_by_top_prob"
    metric = "accuracy"
    keys = ["prompt", "choices", "correct_answer"]
    eval_prompt_key = "prompt"
    eval_answer_key = "choices"

    def __init__(self):
        super().__init__()

    def download(self):
        self.dataset = load_dataset(
            self.path, keep_in_memory=True, trust_remote_code=True
        )

    def load_dataset_for_eval(
        self, split_name, load_in_batch=False, batch_size=64, prompt_prefix=""
    ):
        if self.dataset is None:
            self.download()
        dataset = self.remove_unused_subsets()
        dataset = process_docs(dataset)
        dataset = dataset[split_name]
        dataset = dataset.map(
            lambda x: {
                "prompt": x["query"],
                "choices": [" " + c for c in x["choices"]],
                "correct_answer": x["gold"],
            }
        )
        dataset = dataset.map(lambda x: {"prompt": prompt_prefix + x["prompt"]})
        dataset = dataset.remove_columns(
            [
                "ind",
                "activity_label",
                "ctx_a",
                "ctx_b",
                "ctx",
                "endings",
                "source_id",
                "split",
                "split_type",
                "query",
                "gold",
                "label",
            ]
        )
        return self.batchify(dataset, batch_size) if load_in_batch else dataset
