from datasets import load_dataset

from eco.dataset.base import BaseDataset
from eco.dataset.utils import mmlu_subjects


class MMLU(BaseDataset):
    dataset_type = "multiple_choice"
    path = "cais/mmlu"
    name = "mmlu"
    subjects = mmlu_subjects.copy()
    subsets = ["auxiliary_train", "dev", "validation", "test"]
    test_set = "test"
    choice_labels = ["A", "B", "C", "D"]
    eval_method = "choice_by_top_logit"
    metric = "accuracy"
    keys = ["prompt", "choices", "correct_answer"]
    eval_prompt_key = "prompt"
    eval_answer_key = "choices"

    def __init__(self):
        super().__init__()

    def download(self):
        self.dataset = load_dataset(
            self.path, "all", keep_in_memory=True, trust_remote_code=True
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
                "prompt": MMLU.format_prompt(
                    x["question"], x["choices"], self.choice_labels, x["subject"]
                )
            }
        )
        dataset = dataset.map(lambda x: {"prompt": prompt_prefix + x["prompt"]})
        dataset = dataset.map(lambda x: {"choices": self.choice_labels})
        dataset = dataset.remove_columns(["question", "subject"])
        dataset = dataset.rename_column("answer", "correct_answer")
        return self.batchify(dataset, batch_size) if load_in_batch else dataset

    @staticmethod
    def format_prompt(prompt, choice_text, choice_label, subject):
        subject = subject.replace("_", " ")
        if subject:
            topic_line = (
                f"The following are multiple choice questions (with answers) about "
                + f"{subject}.\n\n"
            )
        else:
            topic_line = (
                f"The following are multiple choice questions (with answers).\n\n"
            )
        question_line = f"{prompt}\n"
        choice_lines = "\n".join(
            [f"{label}. {text}" for label, text in zip(choice_label, choice_text)]
        )
        answer_line = f"\nAnswer:"
        return topic_line + question_line + choice_lines + answer_line
