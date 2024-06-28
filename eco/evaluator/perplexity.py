import evaluate

from eco.evaluator.utils import perplexity


class Perplexity:
    name = "perplexity"

    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def evaluate(self, answers, generated_answers):
        scores = perplexity(
            generated_answers, self.model, self.tokenizer, reduction="none"
        )
        return scores
