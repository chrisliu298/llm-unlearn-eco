import evaluate


class METEOR:
    name = "meteor"

    def __init__(self):
        super().__init__()
        self.scorer = evaluate.load("meteor")

    def evaluate(self, answers, generated_answers):
        scores = []
        for a, ga in zip(answers, generated_answers):
            scores.append(
                self.scorer.compute(predictions=[ga], references=[a])["meteor"]
            )
        return scores
