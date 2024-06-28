from datasets import load_dataset

from eco.dataset.base import BaseDataset


class OpenBookQA(BaseDataset):
    dataset_type = "multiple_choice"
    path = "allenai/openbookqa"
    name = "openbookqa"
    subsets = ["train", "validation", "test"]
    test_set = "test"
    eval_method = "choice_by_top_prob"
    metric = "accuracy"
    keys = ["prompt", "choices", "correct_answer"]
    eval_prompt_key = "prompt"
    eval_answer_key = "choices"

    def __init__(self):
        super().__init__()

    def download(self):
        self.dataset = load_dataset(
            self.path, "main", keep_in_memory=True, trust_remote_code=True
        )

    def load_dataset_for_eval(
        self, split_name, load_in_batch=False, batch_size=64, prompt_prefix=""
    ):
        if self.dataset is None:
            self.download()
        dataset = self.remove_unused_subsets()
        dataset = dataset[split_name]
        dataset = dataset.map(
            lambda x: {
                "prompt": OpenBookQA.format_prompt(x["question_stem"]),
                "choices": [" " + c for c in x["choices"]["text"]],
                "correct_answer": x["choices"]["label"].index(x["answerKey"]),
            }
        )
        dataset = dataset.map(lambda x: {"prompt": prompt_prefix + x["prompt"]})
        dataset = dataset.remove_columns(["id", "question_stem", "answerKey"])
        return self.batchify(dataset, batch_size) if load_in_batch else dataset

    def load_dataset_for_classification(self, split_name):
        return

    @staticmethod
    def format_prompt(prompt):
        return f"Question: {prompt}\nAnswer:"
