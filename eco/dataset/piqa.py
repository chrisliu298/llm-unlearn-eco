from datasets import load_dataset

from eco.dataset.base import BaseDataset


class PIQA(BaseDataset):
    dataset_type = "multiple_choice"
    path = "piqa"
    name = "piqa"
    subsets = ["train", "validation", "test"]
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
                "prompt": PIQA.format_prompt(x["goal"]),
                "choices": [" " + x["sol1"], " " + x["sol2"]],
                "correct_answer": x["label"],
            }
        )
        dataset = dataset.map(lambda x: {"prompt": prompt_prefix + x["prompt"]})
        dataset = dataset.remove_columns(["goal", "sol1", "sol2", "label"])
        return self.batchify(dataset, batch_size) if load_in_batch else dataset

    @staticmethod
    def format_prompt(prompt):
        return prompt
