from typing import Collection, Hashable, Optional

import numpy as np


def _valid_items(row: np.ndarray, ignore_values: Collection[Hashable]) -> set:
    items = set()
    for value in row.tolist():
        if value in ignore_values:
            continue
        try:
            if np.isnan(value):
                continue
        except TypeError:
            pass
        items.add(value)
    return items


def recall_at_k(
    results: np.ndarray,
    ground_truth: np.ndarray,
    k: int,
    *,
    ignore_values: Optional[Collection[Hashable]] = None,
) -> float:
    """Compute recall@k over ranked result ids.

    ``results`` and ``ground_truth`` must contain at least ``k`` ids per query.
    Within those top-k ids, duplicate ids and ignored padding values are counted
    once, and the denominator is the number of valid ground-truth ids.
    """
    results = np.asarray(results)
    ground_truth = np.asarray(ground_truth)

    if results.ndim != 2:
        raise ValueError("results must be a 2D array")
    if ground_truth.ndim != 2:
        raise ValueError("ground_truth must be a 2D array")
    if results.shape[0] != ground_truth.shape[0]:
        raise ValueError(
            "results and ground_truth must have the same number of rows: "
            f"{results.shape[0]} != {ground_truth.shape[0]}"
        )
    if not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an integer")
    if k <= 0:
        raise ValueError("k must be positive")
    if results.shape[1] < k:
        raise ValueError(
            f"results has only {results.shape[1]} columns, cannot compute recall@{k}"
        )
    if ground_truth.shape[1] < k:
        raise ValueError(
            "ground_truth has only "
            f"{ground_truth.shape[1]} columns, cannot compute recall@{k}"
        )

    ignored = set(ignore_values or ())
    hit_count = 0
    relevant_count = 0

    for row_pred, row_true in zip(results[:, :k], ground_truth[:, :k], strict=True):
        predicted = _valid_items(row_pred, ignored)
        relevant = _valid_items(row_true, ignored)
        hit_count += len(predicted & relevant)
        relevant_count += len(relevant)

    if relevant_count == 0:
        return 0.0
    return float(hit_count / relevant_count)
