def select_token_deficit_index(
    token_counts, token_weights, candidate_indexes=None,
):
    """Select the eligible source furthest below its cumulative token target."""
    if len(token_counts) != len(token_weights) or not token_counts:
        raise ValueError("token_counts and token_weights must have the same non-zero length")
    if any(count < 0 for count in token_counts):
        raise ValueError("token counts must be non-negative")
    if any(weight < 0 for weight in token_weights):
        raise ValueError("token weights must be non-negative")

    total_weight = sum(token_weights)
    if total_weight <= 0:
        raise ValueError("at least one token weight must be positive")

    if candidate_indexes is None:
        candidate_indexes = tuple(range(len(token_weights)))
    else:
        candidate_indexes = tuple(candidate_indexes)
        if not candidate_indexes:
            raise ValueError("candidate_indexes must not be empty")
        if len(set(candidate_indexes)) != len(candidate_indexes):
            raise ValueError("candidate_indexes must not contain duplicates")
        if any(
            index < 0 or index >= len(token_weights)
            for index in candidate_indexes
        ):
            raise ValueError("candidate index is out of range")

    total_tokens = sum(token_counts)
    if total_tokens == 0:
        return max(candidate_indexes, key=token_weights.__getitem__)

    target_scale = total_tokens / total_weight
    return max(
        candidate_indexes,
        key=lambda index: (
            token_weights[index] * target_scale - token_counts[index],
            token_weights[index],
            -index,
        ),
    )
