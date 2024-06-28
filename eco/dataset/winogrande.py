from datasets import load_dataset

from eco.dataset.base import BaseDataset


# From https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/winogrande/preprocess_winogrande.py
def doc_to_text(doc):
    answer_to_num = {"1": 0, "2": 1}
    return answer_to_num[doc["answer"]]


# From https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/winogrande/preprocess_winogrande.py
def doc_to_target(doc):
    idx = doc["sentence"].index("_") + 1
    return doc["sentence"][idx:].strip()


# From https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/winogrande/preprocess_winogrande.py
def doc_to_choice(doc):
    idx = doc["sentence"].index("_")
    options = [doc["option1"], doc["option2"]]
    return [doc["sentence"][:idx] + opt for opt in options]


class Winogrande(BaseDataset):
    dataset_type = "multiple_choice"
    path = "winogrande"
    name = "winogrande"
    subsets = ["train", "validation"]
    test_set = "validation"
    eval_method = "choice_by_top_prob_prompt"
    metric = "accuracy"
    keys = ["prompt", "choices", "correct_answer"]
    eval_prompt_key = "prompt"
    eval_answer_key = "choices"

    def __init__(self):
        super().__init__()

    def download(self):
        self.dataset = load_dataset(
            self.path, "winogrande_xl", keep_in_memory=True, trust_remote_code=True
        )

    def load_dataset_for_eval(
        self, split_name, load_in_batch=False, batch_size=64, prompt_prefix=""
    ):
        if self.dataset is None:
            self.download()
        dataset = self.remove_unused_subsets()
        dataset = dataset[split_name]
        answer_to_num = {"1": 0, "2": 1}
        dataset = dataset.map(
            lambda x: {
                "prompt": doc_to_choice(x),
                "choices": " " + doc_to_target(x),
                "correct_answer": answer_to_num[x["answer"]],
            }
        )
        dataset = dataset.map(
            lambda x: {
                "prompt": [
                    prompt_prefix + x["prompt"][0],
                    prompt_prefix + x["prompt"][1],
                ]
            }
        )
        dataset = dataset.remove_columns(["sentence", "option1", "option2"])
        return self.batchify(dataset, batch_size) if load_in_batch else dataset
