import evaluate


class BLEURT:
    name = "bleurt"

    def __init__(self, checkpoint="BLEURT-20"):
        super().__init__()
        self.scorer = evaluate.load("bleurt", checkpoint, module_type="metric")

    def evaluate(self, answers, generated_answers):
        scores = self.scorer.compute(predictions=generated_answers, references=answers)[
            "scores"
        ]
        return scores
