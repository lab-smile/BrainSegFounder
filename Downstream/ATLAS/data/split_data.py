import random
from typing import Optional
from collections.abc import Sized


def get_split_indices(data: Sized, split_fraction: float, seed: Optional[int] = None) -> tuple[list[int], list[int]]:
    if not 0 <= split_fraction <= 1:
        raise ValueError(f"Split fraction must be between 0 and 1, got {split_fraction}")

    if seed is not None:
        random.seed(seed)

    total_indices = len(data)
    indices = list(range(total_indices))
    random.shuffle(indices)

    split_point = int(split_fraction * total_indices)

    return indices[:split_point], indices[split_point:]
