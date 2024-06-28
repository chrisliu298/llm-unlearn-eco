from transformers import AutoTokenizer, logging, pipeline

from eco.attack.utils import match_labeled_tokens, pad_to_same_length

logging.set_verbosity_error()


class Classifier:
    task = None

    def __init__(self, model_name, model_path, batch_size):
        self.model_name = model_name
        self.model_path = model_path
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = pipeline(
            self.task, model=self.model_path, tokenizer=self.tokenizer, device=0
        )

    def predict(self, prompt):
        return self.model(
            prompt,
            truncation=True,
            max_length=512,
            padding="longest",
            batch_size=self.batch_size,
        )


class PromptClassifier(Classifier):
    task = "text-classification"

    def __init__(self, model_name, model_path, batch_size):
        super().__init__(model_name, model_path, batch_size)

    def predict(self, prompt, threshold=0.5):
        preds = self.model(
            prompt,
            truncation=True,
            max_length=512,
            padding="longest",
            batch_size=self.batch_size,
        )
        pred_labels = []
        for pred in preds:
            pred_labels.append(
                1 if pred["label"] == "LABEL_1" and pred["score"] > threshold else 0
            )
        return pred_labels


class TokenClassifier(Classifier):
    task = "token-classification"

    def __init__(self, model_name, model_path, batch_size, condition_fn=lambda x: True):
        super().__init__(model_name, model_path, batch_size)
        self.condition_fn = condition_fn

    def predict(self, prompt):
        return self.model(prompt, batch_size=self.batch_size)

    def predict_target_token_labels(self, prompt, target_tokenizer):
        predictions = self.predict(prompt)
        # Get indices of labeled tokens
        labeled_indices = [
            [d["index"] for d in pred if self.condition_fn(d)] for pred in predictions
        ]
        tokenized_prompts = [
            self.tokenizer(p, return_offsets_mapping=True) for p in prompt
        ]
        # Mark tokens as labeled or not for tokens in all prompts
        token_labels = [
            [
                1 if i in labeled_indices[j] else 0
                for i in range(len(tokenized_prompts[j]["input_ids"]))
            ]
            for j in range(len(prompt))
        ]
        target_tokenized_prompts = [
            target_tokenizer(p, return_offsets_mapping=True) for p in prompt
        ]
        # Convert to the target tokenization
        target_token_labels = [
            match_labeled_tokens(
                token_labels[i],
                tokenized_prompts[i]["offset_mapping"],
                target_tokenized_prompts[i]["offset_mapping"],
            )
            for i in range(len(prompt))
        ]
        # If all tokens are unlabeled, mark all but the last tokens as labeled as a safety measure.
        # Note that this is just an implementation decision and can be changed.
        target_token_labels_processed = []
        for token_labels in target_token_labels:
            if all(label == 0 for label in token_labels):
                target_token_labels_processed.append(
                    [1] * (len(token_labels) - 1) + [0]
                )
            else:
                target_token_labels_processed.append(token_labels)

        target_token_labels_processed = pad_to_same_length(
            target_token_labels_processed, padding_side=target_tokenizer.padding_side
        )
        return target_token_labels_processed
