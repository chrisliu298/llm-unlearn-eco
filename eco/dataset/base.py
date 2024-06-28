from datasets import DatasetDict


class BaseDataset:
    dataset_type = None
    path = None
    name = None
    subsets = []
    test_set = None
    eval_method = None
    metric = None
    keys = []

    def __init__(self, *args, **kwargs):
        self.dataset = None

    def download(self):
        raise NotImplementedError(
            f"download not implemented for {self.__class__.__name__}"
        )

    def batchify(self, dataset, batch_size=64):
        keys = self.keys
        extracted_data = {key: dataset[key] for key in keys}
        batched_dataset = []
        dataset_length = len(extracted_data[keys[0]])
        for i in range(0, dataset_length, batch_size):
            batch = {key: extracted_data[key][i : i + batch_size] for key in keys}
            batched_dataset.append(batch)
        return batched_dataset

    def remove_unused_subsets(self):
        if list(self.dataset.column_names.keys()) != self.subsets:
            return DatasetDict({s: self.dataset[s] for s in self.subsets})
        else:
            return self.dataset

    def load_dataset_for_train(self):
        raise NotImplementedError(
            f"load_dataset_for_train not implemented for {self.__class__.__name__}"
        )

    def load_dataset_for_eval(self):
        raise NotImplementedError(
            f"load_dataset_for_eval not implemented for {self.__class__.__name__}"
        )

    def load_dataset_for_classification(self):
        raise NotImplementedError(
            f"load_dataset_for_classification not implemented for {self.__class__.__name__}"
        )

    @staticmethod
    def format_prompt(prompt):
        raise NotImplementedError(f"format_prompt not implemented")
