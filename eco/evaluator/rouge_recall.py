from rouge_score import rouge_scorer


class ROUGERecall:
    name = "rouge_recall"

    def __init__(self, mode="rougeL"):
        super().__init__()
        self.mode = mode
        self.scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        self.name = f"{mode}_recall"

    def evaluate(self, answers, generated_answers):
        scores = []
        for a, ga in zip(answers, generated_answers):
            scores.append(self.scorer.score(a, ga)[self.mode].recall)
        return scores
