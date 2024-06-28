import torch


@torch.no_grad()
def rand_noise_first_n(data, pos, dims, strength):
    pos_mask = torch.tensor(pos, dtype=torch.bool, device=data.device)
    if not pos_mask.any():
        return data
    indices = torch.where(pos_mask.unsqueeze(-1))
    noise = torch.normal(
        mean=0,
        std=strength,
        size=(indices[0].shape[0], dims),
        device=data.device,
        dtype=data.dtype,
    )
    noise_expanded = torch.zeros(
        (data.shape[0], data.shape[1], dims),
        device=data.device,
        dtype=data.dtype,
    )
    noise_expanded[indices[0], indices[1], :] = noise
    data[:, :, :dims] += noise_expanded
    return data


def rand_noise_rand_n(data, pos, dims, strength):
    pos_mask = torch.tensor(pos, dtype=torch.bool, device=data.device)
    if not pos_mask.any():
        return data
    indices = torch.where(pos_mask.unsqueeze(-1))

    noise = torch.normal(
        mean=0.0,
        std=strength,
        size=(indices[0].shape[0], dims),
        device=data.device,
        dtype=data.dtype,
    )

    noise_expanded = torch.zeros(
        data.shape,
        device=data.device,
        dtype=data.dtype,
    )

    total_dims = data.shape[2]
    rand_dims = torch.randperm(total_dims)[:dims]

    for d in rand_dims:
        noise_expanded[indices[0], indices[1], d] = noise[
            :, rand_dims.tolist().index(d)
        ]

    data += noise_expanded

    return data


@torch.no_grad()
def rand_noise_top_k(data, pos, dims, strength):
    pos_mask = torch.tensor(pos, dtype=torch.bool, device=data.device)
    if not pos_mask.any():
        return data
    expanded_pos_mask = pos_mask.unsqueeze(-1).expand_as(data)
    selected_values = torch.masked_select(data, expanded_pos_mask).view(
        -1, data.size(2)
    )
    _, top_k_indices = selected_values.abs().topk(dims, dim=1)
    noise = torch.normal(
        mean=0, std=strength, size=top_k_indices.size(), device=data.device
    )
    row_indices = torch.arange(top_k_indices.size(0), device=data.device).unsqueeze(1)
    selected_values[row_indices, top_k_indices] += noise
    data[expanded_pos_mask] = selected_values.flatten()
    return data


@torch.no_grad()
def zero_out_top_k(data, pos, dims):
    pos_mask = torch.tensor(pos, dtype=torch.bool, device=data.device)
    if not pos_mask.any():
        return data
    indices = torch.where(pos_mask.unsqueeze(-1))

    # Extract the relevant values where pos_mask is True
    selected_values = data[indices[0], indices[1], :]

    # Find top k values to zero out
    _, top_k_indices = selected_values.abs().topk(dims, dim=1)
    row_indices = torch.arange(top_k_indices.size(0), device=data.device).unsqueeze(1)

    # Zero out top k values
    selected_values[row_indices, top_k_indices] = 0

    # Place zeroed values back into the data tensor
    data[indices[0], indices[1], :] = selected_values

    return data


@torch.no_grad()
def flip_sign_first_n(data, pos, dims):
    pos_mask = torch.tensor(pos, dtype=torch.bool, device=data.device)
    if not pos_mask.any():
        return data
    indices = torch.where(pos_mask.unsqueeze(-1))

    # Selecting the data based on indices for modification
    selected_data = data[indices[0], indices[1], :dims]

    # Flipping the sign of the selected data
    data[indices[0], indices[1], :dims] = -selected_data

    return data


@torch.no_grad()
def flip_sign_top_k(data, pos, dims):
    pos_mask = torch.tensor(pos, dtype=torch.bool, device=data.device)
    if not pos_mask.any():
        return data
    expanded_pos_mask = pos_mask.unsqueeze(-1).expand_as(data)
    selected_values = torch.masked_select(data, expanded_pos_mask).view(
        -1, data.size(2)
    )
    _, top_k_indices = selected_values.abs().topk(dims, dim=1)
    row_indices = torch.arange(top_k_indices.size(0), device=data.device).unsqueeze(1)
    selected_values[row_indices, top_k_indices] *= -1
    data[expanded_pos_mask] = selected_values.flatten()
    return data


@torch.no_grad()
def sub_value_top_k(data, pos, dims, strength):
    pos_mask = torch.tensor(pos, dtype=torch.bool, device=data.device)
    if not pos_mask.any():
        return data
    expanded_pos_mask = pos_mask.unsqueeze(-1).expand_as(data)
    selected_values = torch.masked_select(data, expanded_pos_mask).view(
        -1, data.size(2)
    )
    _, top_k_indices = selected_values.topk(dims, dim=1)
    row_indices = torch.arange(top_k_indices.size(0), device=data.device).unsqueeze(1)
    selected_values[row_indices, top_k_indices] -= strength
    data[expanded_pos_mask] = selected_values.flatten()
    return data


@torch.no_grad()
def add_value_least_k(data, pos, dims, strength):
    pos_mask = torch.tensor(pos, dtype=torch.bool, device=data.device)
    if not pos_mask.any():
        return data
    expanded_pos_mask = pos_mask.unsqueeze(-1).expand_as(data)
    selected_values = torch.masked_select(data, expanded_pos_mask).view(
        -1, data.size(2)
    )
    _, least_k_indices = selected_values.topk(dims, dim=1, largest=False)
    row_indices = torch.arange(least_k_indices.size(0), device=data.device).unsqueeze(1)
    selected_values[row_indices, least_k_indices] += strength
    data[expanded_pos_mask] = selected_values.flatten()
    return data


@torch.no_grad()
def sub_value_first_n(data, pos, dims, strength):
    pos_mask = torch.tensor(pos, dtype=torch.bool, device=data.device)
    if not pos_mask.any():
        return data
    expanded_pos_mask = pos_mask.unsqueeze(-1).expand_as(data)
    dims_mask = torch.zeros_like(data, dtype=torch.bool)
    dims_mask[:, :, :dims] = True
    sub_mask = expanded_pos_mask & dims_mask
    data[sub_mask] -= strength
    return data


@torch.no_grad()
def add_value_first_n(data, pos, dims, strength):
    pos_mask = torch.tensor(pos, dtype=torch.bool, device=data.device)
    if not pos_mask.any():
        return data
    expanded_pos_mask = pos_mask.unsqueeze(-1).expand_as(data)
    dims_mask = torch.zeros_like(data, dtype=torch.bool)
    dims_mask[:, :, :dims] = True
    add_mask = expanded_pos_mask & dims_mask
    data[add_mask] += strength
    return data


@torch.no_grad()
def set_rand_noise_first_n(data, pos, dims, strength):
    # Convert pos to a boolean tensor to create a mask.
    pos_mask = torch.tensor(pos, dtype=torch.bool, device=data.device)
    if not pos_mask.any():
        return data
    indices = torch.where(pos_mask.unsqueeze(-1))
    noise = torch.normal(
        mean=0,
        std=strength,
        size=(indices[0].shape[0], dims),
        device=data.device,
        dtype=data.dtype,
    )
    noise_expanded = torch.zeros(
        (data.shape[0], data.shape[1], dims),
        device=data.device,
        dtype=data.dtype,
    )
    noise_expanded[indices[0], indices[1], :] = noise
    data[:, :, :dims] = noise_expanded
    return data


@torch.no_grad()
def zero_out_first_n(data, pos, dims):
    pos_mask = torch.tensor(pos, dtype=torch.bool, device=data.device)
    if not pos_mask.any():
        return data
    indices = torch.where(pos_mask.unsqueeze(-1))
    data[indices[0], indices[1], :dims] = 0
    return data


def reverse_order(data, *args, **kwargs):
    data = torch.flip(data, [1])
    return data


@torch.no_grad()
def shuffle(data, *args, **kwargs):
    rand_order = torch.randperm(data.size(1), device=data.device)
    data = data[:, rand_order, :]
    return data


corrupt_methods = {
    "rand_noise_first_n": rand_noise_first_n,
    "rand_noise_top_k": rand_noise_top_k,
    "zero_out_top_k": zero_out_top_k,
    "flip_sign_first_n": flip_sign_first_n,
    "flip_sign_top_k": flip_sign_top_k,
    "sub_value_top_k": sub_value_top_k,
    "add_value_least_k": add_value_least_k,
    "set_rand_noise_first_n": set_rand_noise_first_n,
    "rand_noise_rand_n": rand_noise_rand_n,
    "zero_out_first_n": zero_out_first_n,
    "reverse_order": reverse_order,
    "shuffle": shuffle,
    "sub_value_first_n": sub_value_first_n,
    "add_value_first_n": add_value_first_n,
}
