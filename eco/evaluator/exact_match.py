import evaluate


class ExactMatch:
    name = "exact_match"

    def __init__(self):
        super().__init__()
        self.scorer = evaluate.load("exact_match")

    def evaluate(self, answers, generated_answers):
        scores = []
        for a, ga in zip(answers, generated_answers):
            scores.append(
                self.scorer.compute(predictions=[ga], references=[a])["exact_match"]
            )
        return scores
