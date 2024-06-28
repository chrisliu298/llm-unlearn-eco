import torch


class ChoiceByTopLogit:
    name = "choice_by_top_logit"

    def __init__(self, save_logits=False):
        super().__init__()
        self.save_logits = save_logits
        self.logits = []

    def evaluate(self, prompts, answers, model, tokenizer):
        padding_side = tokenizer.padding_side

        inputs = prompts
        prompt_encoding = tokenizer(inputs, padding="longest", return_tensors="pt").to(
            model.device
        )
        choice_encoding = (
            tokenizer(answers[0], return_tensors="pt", add_special_tokens=False)
            .input_ids.to(model.device)
            .squeeze(1)
        )

        with torch.no_grad():
            logits = model(**prompt_encoding, prompts=prompts, answers=None).logits

        # Get the top logit for each choice
        top_logit_choices = []
        for i, (_, attn_mask) in enumerate(
            zip(prompt_encoding["input_ids"], prompt_encoding["attention_mask"])
        ):
            # Find the last token of the prompt and get the logit
            prompt_end = sum(attn_mask) - 1 if padding_side == "right" else -1
            choice = torch.argmax(logits[i, prompt_end, choice_encoding], dim=-1).item()
            top_logit_choices.append(choice)
            if self.save_logits:
                self.logits.append(logits[i, prompt_end, :].view(1, -1).cpu())

        return top_logit_choices
