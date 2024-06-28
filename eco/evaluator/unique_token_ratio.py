import torch


class UniqueTokenRatio:
    name = "unique_token_ratio"

    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def evaluate(self, answers, generated_answers):
        ratios = []
        for ga in generated_answers:
            input_ids = self.tokenizer(
                ga, return_tensors="pt", add_special_tokens=False
            ).input_ids[0]
            if len(input_ids) == 0:
                ratios.append(0)
                continue
            num_unique_tokens = torch.unique(input_ids, sorted=False).shape[0]
            ratio = num_unique_tokens / input_ids.shape[0]
            ratios.append(ratio)
        return ratios
