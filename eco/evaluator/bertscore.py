import evaluate
from transformers import AutoTokenizer


class BERTScore:
    name = "bertscore"

    def __init__(self, mode="f1"):
        super().__init__()
        self.mode = mode
        self.scorer = evaluate.load("bertscore")
        self.name = f"bertscore_{mode}"
        self.bert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def evaluate(self, answers, generated_answers):
        max_length = 500
        answers = self.bert_tokenizer.batch_decode(
            self.bert_tokenizer(
                answers, max_length=max_length, truncation=True
            ).input_ids,
            skip_special_tokens=True,
        )
        generated_answers = self.bert_tokenizer.batch_decode(
            self.bert_tokenizer(
                generated_answers, max_length=max_length, truncation=True
            ).input_ids,
            skip_special_tokens=True,
        )
        scores = self.scorer.compute(
            predictions=generated_answers,
            references=answers,
            lang="en",
            model_type="distilbert-base-uncased",
            batch_size=len(answers),
            device=None,
        )[self.mode]
        return scores
