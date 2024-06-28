from datasets import load_dataset

from eco.dataset.base import BaseDataset


class BoolQ(BaseDataset):
    dataset_type = "multiple_choice"
    path = "google/boolq"
    name = "boolq"
    subsets = ["train", "validation"]
    test_set = "validation"
    choice_labels = ["Yes", "No"]
    eval_method = "choice_by_top_logit"
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
        answer_map = {"True": 0, "False": 1}
        dataset = dataset.map(
            lambda x: {
                "prompt": BoolQ.format_prompt(x["passage"], x["question"]),
                "choices": self.choice_labels,
                "correct_answer": answer_map[str(x["answer"])],
            }
        )
        dataset = dataset.map(lambda x: {"prompt": prompt_prefix + x["prompt"]})
        dataset = dataset.remove_columns(["question", "passage", "answer"])
        return self.batchify(dataset, batch_size) if load_in_batch else dataset

    @staticmethod
    def format_prompt(passage, question):
        return f"{passage}\nQuestion: {question}\nAnswer:"
