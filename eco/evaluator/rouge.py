import evaluate


class ROUGE:
    name = "rouge"

    def __init__(self, mode):
        super().__init__()
        assert mode in [
            "rouge1",
            "rouge2",
            "rougeL",
            "rougeLsum",
        ], f"Invalid mode: {mode}, must be one of rouge1, rouge2, rougeL, rougeLsum"
        self.mode = mode
        self.scorer = evaluate.load("rouge")
        self.name = mode

    def evaluate(self, answers, generated_answers):
        scores = []
        for a, ga in zip(answers, generated_answers):
            scores.append(
                self.scorer.compute(predictions=[ga], references=[a])[self.mode]
            )
        return scores
