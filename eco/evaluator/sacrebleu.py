import evaluate


class SacreBLEU:
    name = "sacrebleu"

    def __init__(self, mode="token"):
        super().__init__()
        assert mode in [
            "token",
            "sentence",
        ], f"Invalid mode: {mode}, choose from 'token' or 'sentence'"
        self.sentence_level = True if mode == "sentence" else False
        self.scorer = evaluate.load("sacrebleu")
        self.name = f"sacrebleu_{mode}"

    def evaluate(self, answers, generated_answers):
        scores = []
        for a, ga in zip(answers, generated_answers):
            scores.append(
                self.scorer.compute(
                    predictions=[ga],
                    references=[a],
                    use_effective_order=self.sentence_level,
                )["score"]
                / 100
            )
        return scores
