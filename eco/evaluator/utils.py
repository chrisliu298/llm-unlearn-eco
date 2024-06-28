import numpy as np
import torch
from scipy.special import logsumexp


def answer_prob(prompts, answers, model, tokenizer, reduction="mean"):
    padding_side = tokenizer.padding_side

    # Concatenate each question and answer pair and encode them
    inputs = [p + a for p, a in zip(prompts, answers)]
    encoding = tokenizer(inputs, padding="longest", return_tensors="pt").to(
        model.device
    )

    # Get model's output (logits) for the batch
    with torch.no_grad():  # Disable gradient calculation for inference
        logits = model(
            **encoding, labels=encoding["input_ids"], prompts=prompts, answers=answers
        ).logits

    # Shift logits and labels to align for calculating the probability of answer tokens
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = encoding["input_ids"][..., 1:].contiguous()

    # Flatten the logits and labels to calculate loss easily across the batch
    flatten_logits = shift_logits.view(-1, shift_logits.size(-1))
    flatten_labels = shift_labels.view(-1)

    # Calculate loss using CrossEntropy to get log probabilities, then negate for actual log probs
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(flatten_logits, flatten_labels) * -1
    loss = loss.view(shift_labels.size())

    answer_log_probs = []
    for i, (_, attn_mask) in enumerate(
        zip(encoding["input_ids"], encoding["attention_mask"])
    ):
        # Find the start index of the answer and its actual length
        if padding_side == "right":
            answer_start = len(tokenizer.encode(prompts[i])) - 1
            answer_end = sum(attn_mask) - 1
            answer_length = answer_end - answer_start
        else:
            answer_start = (
                (attn_mask == 0).sum() + len(tokenizer.encode(prompts[i])) - 1
            )
            answer_end = len(encoding["input_ids"][i])
            answer_length = answer_end - answer_start

        # Select log probabilities corresponding to the actual answer tokens
        answer_log_probs.append(loss[i, answer_start : answer_start + answer_length])

    if reduction == "mean":
        answer_log_probs = [log_probs.mean().cpu() for log_probs in answer_log_probs]
    elif reduction == "sum":
        answer_log_probs = [log_probs.sum().cpu() for log_probs in answer_log_probs]
    else:
        answer_log_probs = [log_probs.cpu() for log_probs in answer_log_probs]

    return answer_log_probs


def normalized_prob(prob_answer, all_probs):
    return prob_answer / np.sum(all_probs)


def log_truth_ratio(log_probs_perturbed, log_prob_paraphrased):
    return np.exp(np.array(log_probs_perturbed).mean() - log_prob_paraphrased)


def perplexity(text_batch, model, tokenizer, reduction="mean"):
    # Encode the batch of text inputs
    encoding = tokenizer(text_batch, padding=True, return_tensors="pt").to(model.device)
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=input_ids
        )

    # Calculate loss
    loss_fct = torch.nn.CrossEntropyLoss(
        reduction="none", ignore_index=tokenizer.pad_token_id
    )
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    flatten_logits = shift_logits.view(-1, shift_logits.size(-1))
    flatten_labels = shift_labels.view(-1)
    losses = loss_fct(flatten_logits, flatten_labels)
    losses = losses.view(shift_labels.size())

    avg_losses = torch.mean(losses, dim=1)

    if reduction == "mean":
        perplexity = torch.mean(avg_losses).exp()
    else:
        perplexity = torch.exp(avg_losses)
    return perplexity.cpu().tolist()
