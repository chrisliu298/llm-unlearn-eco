import evaluate


class BLEU:
    name = "bleu"

    def __init__(self):
        super().__init__()
        self.scorer = evaluate.load("bleu")

    def evaluate(self, answers, generated_answers):
        scores = []
        for a, ga in zip(answers, generated_answers):
            scores.append(self.scorer.compute(predictions=[ga], references=[a])["bleu"])
        return scores
