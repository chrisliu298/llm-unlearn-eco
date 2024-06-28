import math

import torch

from eco.evaluator.utils import answer_prob


class AnswerProb:
    name = "answer_prob"

    def __init__(self, to_prob=False):
        super().__init__()
        self.to_prob = to_prob

    def evaluate(self, prompts, answers, model, tokenizer):
        probs = torch.stack(
            answer_prob(prompts, answers, model, tokenizer, "mean")
        ).tolist()
        if self.to_prob:
            probs = [math.exp(p) for p in probs]
        return probs
