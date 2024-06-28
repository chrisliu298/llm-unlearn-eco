import numpy as np
import torch

from eco.evaluator.utils import answer_prob, normalized_prob


class NormalizedAnswerProb:
    name = "normalized_answer_prob"

    def __init__(self):
        super().__init__()

    def evaluate(self, prompts, answers, model, tokenizer):
        normalized_probs = []
        for prompt, answer in zip(prompts, answers):
            prompt = [prompt] * len(answer)
            log_probs = answer_prob(prompt, answer, model, tokenizer, "mean")
            log_probs = torch.stack(log_probs).tolist()
            normalized_probs.append(
                normalized_prob(np.exp(log_probs[0]), np.exp(log_probs))
            )
        return normalized_probs
