# cython: language_level=3

from typing import List, Dict, Tuple, Optional

cimport cython
import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
ctypedef np.int32_t DTYPE_t2

from numpy.math cimport INFINITY


@cython.boundscheck(False) # turn off bounds-checking for entire function
cpdef play_vectorized(
    np.ndarray[DTYPE_t, ndim=3] y_valid,
    np.ndarray[DTYPE_t, ndim=3] y_test,
    np.ndarray[DTYPE_t, ndim=3] runtimes,
    configurations: List[int],
    config_id_to_idx: Dict[int, str],
    np.ndarray[DTYPE_t, ndim=1] cutoffs,
    max_runtime: float,
    cache: Optional[Dict] = None,
    n_jobs: int = 1,
):
    cdef int idx
    cdef int n_tasks = len(y_valid)
    cdef np.ndarray[DTYPE_t2, ndim=1] configuration_indices_array
    cdef np.ndarray[DTYPE_t, ndim=1] time_elapsed
    cdef np.ndarray[DTYPE_t, ndim=1] min_val_score
    cdef np.ndarray[DTYPE_t, ndim=1] min_test_score
    cdef np.ndarray keep
    # There's no convenient way to define boolean arrays
    cdef np.ndarray finites
    cdef np.ndarray[DTYPE_t, ndim=1] lc
    cdef np.ndarray[DTYPE_t, ndim=1] lc_test
    cdef np.ndarray[DTYPE_t, ndim=1] lc_runtime
    cdef int skip_n_configs
    cdef double cutoff
    cdef double c
    cdef double runtime
    cdef int config_idx
    cdef double loss = INFINITY
    cdef double test_loss = INFINITY
    cdef int early_stopping_idx
    cdef int loc

    if cache is None:
        time_elapsed = np.zeros((n_tasks, ), dtype=np.float64)
        min_val_score = np.ones((n_tasks,), dtype=np.float64) * INFINITY
        min_test_score = np.ones((n_tasks,), dtype=np.float64) * INFINITY
        skip_n_configs = 0
    else:
         time_elapsed = cache['time_elapsed'].copy()
         min_val_score = cache['min_val_score'].copy()
         min_test_score = cache['min_test_score'].copy()
         skip_n_configs = cache['skip_n_configs']

    n_tasks = len(y_valid)
    n_iterations = len(configurations)
    configuration_indices = list()
    for member in configurations:
        configuration_indices.append(config_id_to_idx[member])
    configuration_indices_array = np.array(configuration_indices, dtype=np.int32)

    for task_idx in range(n_tasks):

        finites = np.isfinite(
            y_valid[task_idx, configuration_indices_array]
        ).any(axis=1)

        for i in range(skip_n_configs, n_iterations):

            config_idx = configuration_indices_array[i]
            cutoff = cutoffs[i]

            if time_elapsed[task_idx] > max_runtime:
                break

            if not finites[i]:
                runtime = np.nanmax(runtimes[task_idx, config_idx])
                if runtime > cutoff:
                    time_elapsed[task_idx] += cutoff
                else:
                    time_elapsed[task_idx] += runtime
                loss = 1.0
                test_loss = 1.0

            else:

                # This code is way less involved than the successive halving
                # variant - however, it does not require any further speedups
                # because it is called way less often and is therefore not the
                # bottleneck
                rt = runtimes[task_idx, config_idx]
                keep = np.isfinite(rt)
                lc = y_valid[task_idx, config_idx][keep]
                lc_test = y_test[task_idx, config_idx][keep]
                lc_runtime = runtimes[task_idx, config_idx][keep]

                c = min(cutoff, max_runtime - time_elapsed[task_idx])

                if c > lc_runtime[-1]:
                    loss = lc[-1]
                    test_loss = lc_test[-1]
                    time_elapsed[task_idx] += lc_runtime[-1]
                elif c < lc_runtime[0]:
                    time_elapsed[task_idx] += c
                    loss = 1.0
                    test_loss = 1.0
                else:
                    loc = len(lc_runtime[(lc_runtime < c) == True])
                    lc = lc[:loc]
                    lc_test = lc_test[:loc]
                    loss = lc[-1]
                    test_loss = lc_test[-1]
                    time_elapsed[task_idx] += c

            if loss < min_val_score[task_idx]:
                min_val_score[task_idx] = loss
                min_test_score[task_idx] = test_loss

            if time_elapsed[task_idx] > max_runtime:
                break

        if min_test_score[task_idx] != min_test_score[task_idx] or min_test_score[task_idx] == INFINITY:
            min_test_score[task_idx] = 1.0

    cache = dict()
    cache['time_elapsed'] = time_elapsed
    cache['min_val_score'] = min_val_score
    cache['min_test_score'] = min_test_score
    cache['skip_n_configs'] = len(configurations)

    return min_val_score, min_test_score, cache
