from datasets import load_dataset

from eco.dataset.base import BaseDataset


class ARC(BaseDataset):
    dataset_type = "multiple_choice"
    path = "allenai/ai2_arc"
    subsets = ["train", "validation", "test"]
    test_set = "test"
    eval_method = "choice_by_top_prob"
    metric = "accuracy"
    keys = ["prompt", "choices", "correct_answer"]
    eval_prompt_key = "prompt"
    eval_answer_key = "choices"

    def __init__(self):
        super().__init__()

    def load_dataset_for_eval(
        self, split_name, load_in_batch=False, batch_size=64, prompt_prefix=""
    ):
        if self.dataset is None:
            self.download()
        dataset = self.remove_unused_subsets()
        dataset = dataset[split_name]
        dataset = dataset.map(
            lambda x: {
                "prompt": ARC.format_prompt(x["question"]),
                "choices": [" " + c for c in x["choices"]["text"]],
                "correct_answer": x["choices"]["label"].index(x["answerKey"]),
            }
        )
        dataset = dataset.map(lambda x: {"prompt": prompt_prefix + x["prompt"]})
        dataset = dataset.remove_columns(["id", "question", "answerKey"])
        return self.batchify(dataset, batch_size) if load_in_batch else dataset

    @staticmethod
    def format_prompt(prompt):
        return f"Question: {prompt}\nAnswer:"


class ARCEasy(ARC):
    name = "arc-easy"

    def __init__(self):
        super().__init__()

    def download(self):
        self.dataset = load_dataset(
            self.path, "ARC-Easy", keep_in_memory=True, trust_remote_code=True
        )


class ARCChallenge(ARC):
    name = "arc-challenge"

    def __init__(self):
        super().__init__()

    def download(self):
        self.dataset = load_dataset(
            self.path, "ARC-Challenge", keep_in_memory=True, trust_remote_code=True
        )
