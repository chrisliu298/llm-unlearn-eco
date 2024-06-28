from collections import OrderedDict

import torch

from eco.attack.corrupt import corrupt_methods


def apply_corruption_hook(module, corrupt_method, corrupt_args):
    corrupt_fn = corrupt_methods[corrupt_method]

    def corrupt(module, inputs, outputs):
        if outputs.shape[1] > 1:
            outputs = corrupt_fn(outputs, **corrupt_args)
        return outputs

    handle = module.register_forward_hook(corrupt)
    return handle


def apply_embeddings_extraction_hook(module, embeddings):
    def extract_embeddings(module, inputs, outputs):
        embeddings.append(outputs.detach())

    handle = module.register_forward_hook(extract_embeddings)
    return handle


def cosine_similarity_matrix(row_vectors, matrix, eps=1e-8):
    dot_product = torch.mm(row_vectors, matrix.t())
    row_vectors_norm = torch.norm(row_vectors, p=2, dim=1, keepdim=True) + eps
    matrix_norms = torch.norm(matrix, p=2, dim=1, keepdim=True) + eps
    cosine_similarity = dot_product / (row_vectors_norm * matrix_norms.t())
    return cosine_similarity


def embedding_to_tokens(embeddings, embedding_matrix):
    similarities = cosine_similarity_matrix(embeddings, embedding_matrix)
    selected_tokens = torch.argmax(similarities, dim=1)
    return selected_tokens, similarities


def pad_to_same_length(pos, padding_side="right"):
    assert padding_side in [
        "right",
        "left",
    ], "padding_side must be either right or left"
    max_len = max(pos, key=len)
    if padding_side == "right":
        return [i + [0] * (len(max_len) - len(i)) for i in pos]
    else:
        return [[0] * (len(max_len) - len(i)) + i for i in pos]


def match_labeled_tokens(src_labels, src_offsets, tgt_offsets):
    src_target_offsets = [
        offset for offset, label in zip(src_offsets, src_labels) if label == 1
    ]
    tgt_matched_tokens_indices = []
    for i, (tgt_start, tgt_end) in enumerate(tgt_offsets):
        for src_start, src_end in src_target_offsets:
            if src_start < tgt_end and src_end > tgt_start:
                tgt_matched_tokens_indices.append(i)
                break

    tgt_labels = []
    for i in range(len(tgt_offsets)):
        if i in tgt_matched_tokens_indices:
            tgt_labels.append(1)
        else:
            tgt_labels.append(0)
    return tgt_labels


def remove_hooks(model):
    for module in model.modules():
        module._forward_hooks = OrderedDict()


def print_hooks(model):
    for module in model.modules():
        if module._forward_hooks != OrderedDict():
            print(module, module._forward_hooks)


def get_nested_attr(obj, attr):
    for a in attr.split("."):
        obj = getattr(obj, a)
    return obj


def remove_none_values(d):
    return {k: v for k, v in d.items() if v is not None}


def idx_to_mask(idx, length):
    idx_set = set(idx)  # Convert to set for efficient lookup
    return [1 if i in idx_set else 0 for i in range(length)]


def mask_to_idx(mask):
    return [i for i, m in enumerate(mask) if m == 1]
