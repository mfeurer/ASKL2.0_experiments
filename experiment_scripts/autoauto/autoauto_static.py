from collections import defaultdict
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import sklearn.dummy

sys.path.append('..')
import portfolio.portfolio_util


def build(
    performance_matrix: pd.DataFrame,
    matrices: Dict[str, Dict[str, np.ndarray]],
    metafeatures: pd.DataFrame,
    random_state: np.random.RandomState,
    task_ids: List[int],
    configurations: Dict[str, List[str]],
    seed: int,
):
    if performance_matrix.shape[1] != len(matrices):
        raise ValueError('mismatch in the data structure. Performance_matrix vs matrics contain '
                         'the following keys', performance_matrix.columns, matrices.keys())

    strategies = list(matrices.keys())

    minima_for_methods = dict()
    maxima_for_methods = dict()

    for method in strategies:
        _, y_test, _, _, _ = portfolio.portfolio_util.reformat_data(
            matrices[method], task_ids, configurations[method])
        matrix = y_test.copy()
        minima = np.nanmin(np.nanmin(matrix, axis=2), axis=1)
        minima_as_dicts = {
            task_id: minima[i] for i, task_id in enumerate(performance_matrix.index)
        }
        maxima = np.nanmax(np.nanmax(matrix, axis=2), axis=1)
        maxima_as_dicts = {
            task_id: maxima[i] for i, task_id in enumerate(performance_matrix.index)
        }
        minima_for_methods[method] = minima_as_dicts
        maxima_for_methods[method] = maxima_as_dicts
        diff = maxima - minima
        diff[diff == 0] = 1
        for task_idx in range(matrix.shape[0]):
            matrix[task_idx] = (
                (matrix[task_idx] - minima[task_idx]) / diff[task_idx]
            )
        assert (
            np.all((matrix >= 0) | (~np.isfinite(matrix)))
            and np.all((matrix <= 1) | (~np.isfinite(matrix)))
        ), (
            matrix, (matrix >= 0) | (~np.isfinite(matrix))
        )

    # Classification approach - generate data
    y_values = []
    task_id_to_idx = {}
    for i, task_id in enumerate(performance_matrix.index):
        values = []
        task_id_to_idx[task_id] = len(y_values)
        for method in strategies:
            val = performance_matrix[method][task_id]
            values.append(val)
        y_values.append(values)

    # Normalize each column given the minimum and maximum ever observed on these tasks
    y_values = np.array(y_values, dtype=float)
    for task_id in performance_matrix.index:
        task_idx = task_id_to_idx[task_id]
        minima = np.inf
        maxima = -np.inf
        for method in strategies:
            minima = min(minima_for_methods[method][task_id], minima)
            maxima = max(maxima_for_methods[method][task_id], maxima)
        diff = maxima - minima
        if diff == 0:
            diff = 1
        y_values[task_idx] = (y_values[task_idx] - minima) / diff
    mean_normalized_regret = np.mean(y_values, axis=0)
    selection = int(np.argmin(mean_normalized_regret))
    for strategy, mnr in zip(strategies, mean_normalized_regret):
        print(strategy, mnr)
    assert len(mean_normalized_regret) == len(strategies), \
        (len(mean_normalized_regret), len(strategies))
    print(list(strategies)[selection], selection)

    constant = np.array([
            1 if i == selection else 0 for i in range(len(mean_normalized_regret))
        ])
    selector = sklearn.dummy.DummyClassifier(strategy='constant', constant=constant)
    selector.fit(metafeatures, np.tile(constant, (len(metafeatures), 1)))

    return selector
