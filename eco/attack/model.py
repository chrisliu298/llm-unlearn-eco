from eco.attack.utils import (
    apply_corruption_hook,
    apply_embeddings_extraction_hook,
    get_nested_attr,
    pad_to_same_length,
    remove_hooks,
    remove_none_values,
)


class AttackedModel:
    def __init__(
        self,
        model,
        prompt_classifier,
        token_classifier,
        corrupt_method,
        corrupt_args,
        classifier_threshold=0.5,
    ):
        self.model_name = model.model_name
        self.model = model.model
        self.tokenizer = model.tokenizer
        self.model_config = model.model_config
        self.device = model.device
        self.prompt_classifier = prompt_classifier
        self.token_classifier = token_classifier
        self.corrupt_method = corrupt_method
        self.corrupt_args = remove_none_values(corrupt_args)
        self.attack_module = get_nested_attr(
            model.model, model.model_config["attack_module"]
        )
        self.classifier_threshold = classifier_threshold
        self.generation_config = model.generation_config
        self.embeddings_data = []

    def __call__(self, prompts, answers, *args, **kwargs):
        # Prevent error caused by token_type_ids for OLMo-7B-Instruct model
        if (
            "olmo" in self.model_name.lower()
            or "qwen" in self.model_name.lower()
            or self.model_name == "falcon-180B-chat"
        ):
            kwargs.pop("token_type_ids", None)
        remove_hooks(self.model)
        self.apply_corruption(prompts, answers)
        return self.model(*args, **kwargs)

    def update_corrupt_args(self, corrupt_args):
        self.corrupt_args = remove_none_values(corrupt_args)

    def generate(self, prompts, *args, **kwargs):
        # Prevent error caused by token_type_ids for OLMo-7B-Instruct model
        if (
            "olmo" in self.model_name.lower()
            or "qwen" in self.model_name.lower()
            or self.model_name == "falcon-180B-chat"
        ):
            kwargs.pop("token_type_ids", None)
        remove_hooks(self.model)
        self.apply_corruption(prompts)
        return self.model.generate(*args, **kwargs)

    def apply_corruption(self, prompt, answers=None):
        if self.prompt_classifier is not None:
            prompt_attack_label = self.predict_prompt_attack_label(prompt)
        else:
            prompt_attack_label = [1] * len(prompt)
        if answers is not None:
            token_attack_label = self.predict_token_attack_label(
                [p + a for p, a in zip(prompt, answers)]
            )
        else:
            token_attack_label = self.predict_token_attack_label(prompt)
        corruption_pattern = self.make_corruption_pattern(
            prompt_attack_label, token_attack_label
        )
        corrupt_args = self.corrupt_args.copy()
        corrupt_args["pos"] = corruption_pattern.copy()
        corrupt_hook = apply_corruption_hook(
            self.attack_module, self.corrupt_method, corrupt_args
        )
        return corrupt_hook

    def apply_embeddings_extraction(self):
        return apply_embeddings_extraction_hook(self.model, self.embeddings_data)

    def make_corruption_pattern(self, prompt_label, token_label):
        filter_token_labels = []
        for pl, tl in zip(prompt_label, token_label):
            if pl == 1:
                filter_token_labels.append(tl)
            else:
                filter_token_labels.append([0] * len(tl))
        return filter_token_labels

    def predict_prompt_attack_label(self, prompt):
        if "formatting_tokens" in self.model_config:
            prompt_prefix = self.model_config["formatting_tokens"]["prompt_prefix"]
            prompt_suffix = self.model_config["formatting_tokens"]["prompt_suffix"]
            raw_prompt = []
            for p in prompt:
                if p.startswith(prompt_prefix) and p.endswith(prompt_suffix):
                    raw_prompt.append(p[len(prompt_prefix) : -len(prompt_suffix)])
                else:
                    raw_prompt.append(p)
        else:
            raw_prompt = prompt
        return self.prompt_classifier.predict(raw_prompt, self.classifier_threshold)

    def predict_token_attack_label(self, prompt):
        if self.token_classifier is None:
            # Label all but the last token as 1
            tokenized_prompt = [self.tokenizer(p).input_ids for p in prompt]
            token_labels = pad_to_same_length(
                [[1] * (len(p) - 1) + [0] for p in tokenized_prompt],
                padding_side=self.tokenizer.padding_side,
            )
            return token_labels
        token_labels = self.token_classifier.predict_target_token_labels(
            prompt, self.tokenizer
        )
        return token_labels
