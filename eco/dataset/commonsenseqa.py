from datasets import load_dataset

from eco.dataset.base import BaseDataset


class CommonsenseQA(BaseDataset):
    dataset_type = "multiple_choice"
    path = "tau/commonsense_qa"
    name = "commonsenseqa"
    subsets = ["train", "validation"]
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
        dataset = dataset[split_name]
        dataset = dataset.map(
            lambda x: {
                "prompt": CommonsenseQA.format_prompt(x["question"]),
                "choices": [" " + c for c in x["choices"]["text"]],
                "correct_answer": x["choices"]["label"].index(x["answerKey"]),
            }
        )
        dataset = dataset.map(lambda x: {"prompt": prompt_prefix + x["prompt"]})
        dataset = dataset.remove_columns(
            ["id", "question", "answerKey", "question_concept"]
        )
        return self.batchify(dataset, batch_size) if load_in_batch else dataset

    def load_dataset_for_classification(self, split_name):
        return

    @staticmethod
    def format_prompt(prompt):
        return f"Question: {prompt}\nAnswer:"
