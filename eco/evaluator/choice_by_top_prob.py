import torch

from eco.evaluator.utils import answer_prob


class ChoiceByTopProb:
    name = "choice_by_top_prob"

    def __init__(self):
        super().__init__()

    def evaluate(self, prompts, answers, model, tokenizer):
        top_prob_choices = []
        for prompt, choice in zip(prompts, answers):
            if isinstance(prompt, str) and isinstance(choice, list):
                prompt = [prompt] * len(choice)
            elif isinstance(prompt, list) and isinstance(choice, str):
                choice = [choice] * len(prompt)
            else:
                raise ValueError(
                    f"Prompt has type {type(prompt)} and choice has type {type(choice)}\n"
                    + f"Prompt: {prompt}\n"
                    + f"Choice: {choice}"
                )
            log_probs = answer_prob(prompt, choice, model, tokenizer, "mean")
            top_prob_choices.append(torch.argmax(torch.stack(log_probs)).item())
        return top_prob_choices
