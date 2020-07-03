from collections import defaultdict
import copy
import itertools
import json
import os
import pickle
from typing import Dict, Any, List

import numpy as np
import pandas as pd


_training_task_ids = [
    232, 236, 241, 245, 253, 254, 256, 258, 260, 262, 267, 271, 273, 275, 279, 288, 336, 340, 2119,
    2120, 2121, 2122, 2123, 2125, 2356, 3044, 3047, 3048, 3049, 3053, 3054, 3055, 75089, 75092,
    75093, 75098, 75100, 75108, 75109, 75112, 75114, 75115, 75116, 75118, 75120, 75121, 75125,
    75126, 75129, 75131, 75133, 75134, 75136, 75139, 75141, 75142, 75143, 75146, 75147, 75148,
    75149, 75153, 75154, 75156, 75157, 75159, 75161, 75163, 75166, 75169, 75171, 75173, 75174,
    75176, 75178, 75179, 75180, 75184, 75185, 75187, 75192, 75195, 75196, 75199, 75210, 75212,
    75213, 75215, 75217, 75219, 75221, 75223, 75225, 75232, 75233, 75234, 75235, 75236, 75237,
    75239, 75250, 126021, 126024, 126028, 126030, 126031, 146574, 146575, 146576, 146577, 146578,
    146583, 146586, 146592, 146593, 146594, 146596, 146597, 146600, 146601, 146602, 146603,
    146679, 166859, 166866, 166872, 166875, 166882, 166897, 166905, 166906, 166913, 166915, 166931,
    166932, 166944, 166950, 166951, 166953, 166956, 166957, 166958, 166959, 166970, 166996, 167085,
    167086, 167087, 167088, 167089, 167090, 167094, 167096, 167097, 167099, 167100, 167101, 167103,
    167105, 167106, 167202, 167203, 167204, 167205, 168785, 168791, 189779, 189786, 189828, 189829,
    189836, 189840, 189841, 189843, 189844, 189845, 189846, 189857, 189858, 189859, 189863, 189864,
    189869, 189870, 189875, 189878, 189880, 189881, 189882, 189883, 189884, 189887, 189890, 189893,
    189894, 189899, 189900, 189902, 190154, 190155, 190156, 190157, 190158, 190159, 211720, 211721,
    211722, 211723, 211724,
]


def reformat_data(
    matrix: Dict[int, pd.DataFrame],
    task_ids: List[int],
    configurations: List[int],
):
    lc_length = 1
    for task_id in task_ids:
        n_lc_for_task = 0
        for config_id in configurations:
            try:
                if 'test_learning_curve' in matrix[task_id].loc[config_id]['additional_run_info']:
                    tmp_lc_length = len(matrix[task_id].loc[config_id][
                        'additional_run_info'][
                        'test_learning_curve'])
                    lc_length = max(lc_length, tmp_lc_length)
                    n_lc_for_task += 1
            except KeyError as e:

                print(
                    task_id, type(task_id), config_id, type(config_id),
                    matrix[task_id].index, matrix[task_id].columns,
                )
                raise e

    print('Maximal learning curve length', lc_length)
    y_valid = np.ones((len(task_ids), len(configurations), lc_length)) * np.NaN
    y_test = np.ones(y_valid.shape) * np.NaN
    runtimes = np.ones(y_valid.shape) * np.NaN
    config_id_to_idx = dict()
    task_id_to_idx = dict()
    for j, task_id in enumerate(task_ids):
        for k, config_id in enumerate(configurations):
            task_id_to_idx[task_id] = j
            config_id_to_idx[config_id] = k

            mmtc = matrix[task_id].loc[config_id]
            if 'error' in mmtc['additional_run_info']:
                y_valid[j][k][:] = 1.0
                y_test[j][k][:] = 1.0
                runtimes[j][k][:] = mmtc['runtime']
            elif 'test_learning_curve' in mmtc['additional_run_info']:
                lc_length = len(mmtc['additional_run_info']['learning_curve'])
                y_valid[j][k][:lc_length] = mmtc['additional_run_info']['learning_curve']
                y_test[j][k][:lc_length] = mmtc['additional_run_info']['test_learning_curve']
                runtimes[j][k][:lc_length] = mmtc['additional_run_info'][
                    'learning_curve_runtime']
                if lc_length < y_valid.shape[2]:
                    y_valid[j][k][lc_length:] = y_valid[j, k, -1]
                    y_test[j][k][lc_length:] = y_test[j, k, -1]
                    runtimes[j][k][lc_length:] = runtimes[j, k, -1]
            else:
                y_valid[j][k][0] = mmtc['loss']
                y_test[j][k][0] = mmtc['additional_run_info']['test_loss']
                runtimes[j][k][0] = mmtc['runtime']
    return y_valid, y_test, runtimes, config_id_to_idx, task_id_to_idx


def normalize_matrix(matrix):
    normalized_matrix = matrix.copy()
    minima = np.nanmin(np.nanmin(normalized_matrix, axis=2), axis=1)
    maxima = np.nanmax(np.nanmax(normalized_matrix, axis=2), axis=1)
    diff = maxima - minima
    diff[diff == 0] = 1
    for task_idx in range(normalized_matrix.shape[0]):
        normalized_matrix[task_idx] = (
                (normalized_matrix[task_idx] - minima[task_idx]) / diff[task_idx]
        )

    assert (
        np.all((normalized_matrix >= 0) | (~np.isfinite(normalized_matrix)))
        and np.all((normalized_matrix <= 1) | (~np.isfinite(normalized_matrix)))
    ), (
        normalized_matrix, (normalized_matrix >= 0) | (~np.isfinite(normalized_matrix))
    )
    return normalized_matrix
