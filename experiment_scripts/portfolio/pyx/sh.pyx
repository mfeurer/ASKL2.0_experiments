# cython: language_level=3

from typing import List, Dict, Tuple, Optional

cimport cython
import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
ctypedef np.int32_t DTYPE_t2
ctypedef np.int64_t DTYPE_t3

from numpy.math cimport INFINITY


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.initializedcheck(False)  # Turn off check that memory views are initialized
@cython.cdivision(False)  # Turn off checks that division works like in python
cpdef play_vectorized(
    np.ndarray[DTYPE_t, ndim=3] y_valid,
    np.ndarray[DTYPE_t, ndim=3] y_test,
    np.ndarray[DTYPE_t, ndim=3] runtimes,
    configurations: List[str],
    config_id_to_idx: Dict[int, str],
    np.ndarray[DTYPE_t, ndim=1] cutoffs,
    max_runtime: float,
    config_to_budget_to_idx: Dict[str, Dict[float, int]],
    # Successive halving hyperparameters
    int[:] N_i,
    int s_max,
    double[:] R_i,
    # Optional
    cache: Optional[Dict] = None,
    n_jobs: int = 1,
):
    # Only create memviews for y_test and runtimes to still have convenient access to the array
    # methods for y_valid
    cdef double[:,:,:] y_test_memview = y_test
    cdef double[:,:,:] runtimes_memview = runtimes
    cdef int task_idx
    cdef int n_tasks = len(y_valid)
    cdef np.ndarray[DTYPE_t, ndim=1] cutoffs_for_task
    cdef np.ndarray[DTYPE_t, ndim=1] cutoffs_to_race
    cdef np.ndarray[DTYPE_t2, ndim=1] configuration_indices_array
    cdef np.ndarray[DTYPE_t2, ndim=1] configuration_indices_array_for_task
    cdef np.ndarray[DTYPE_t2, ndim=1] configurations_to_race
    cdef double[:] time_elapsed
    cdef double[:] min_val_score
    cdef double[:] min_test_score
    # There's no convenient way to define boolean arrays
    cdef np.ndarray finites
    # Do not make this a typed memory view by rewriting argmin in cython
    # using numpy functions here makes the code more convenient and readable
    cdef np.ndarray[DTYPE_t, ndim=1] losses
    cdef np.ndarray[DTYPE_t3, ndim=1] argsort
    cdef double cutoff
    cdef double c
    cdef double runtime
    cdef int config_idx
    cdef double loss = INFINITY
    cdef double test_loss = INFINITY
    cdef double largest_runtime
    cdef int early_stopping_idx
    cdef int n_configurations_to_race
    cdef int max_budget
    cdef int max_budget_for_algo
    cdef int i
    cdef int j
    cdef int k

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

    n_iterations = len(configurations)
    configuration_indices = list()
    for member in configurations:
        configuration_indices.append(config_id_to_idx[member])
    configuration_indices_array = np.array(configuration_indices, dtype=np.int32)

    for task_idx in range(n_tasks):

        finites = np.isfinite(
            y_valid[task_idx, configuration_indices_array]
        ).any(axis=1)

        configuration_indices_array_for_task = configuration_indices_array.copy()
        cutoffs_for_task = cutoffs.copy()

        n_skipped = 0
        while len(configuration_indices_array_for_task) > 0:
            n_i = N_i[0]
            configurations_to_race = configuration_indices_array_for_task[:n_i]
            configuration_indices_array_for_task = configuration_indices_array_for_task[n_i:]
            cutoffs_to_race = cutoffs_for_task[:n_i]
            cutoffs_for_task = cutoffs_for_task[n_i:]

            if skip_n_configs > 0:
                if n_skipped < skip_n_configs:
                    n_skipped += len(configurations_to_race)
                    continue

            # This here is one iteration of successive halving!
            for i in range(s_max + 1):
                n_configurations_to_race = len(configurations_to_race)

                if i < s_max and N_i[i + 1] >= n_configurations_to_race:
                    # Do nothing if the next iteration expects more
                    # configurations than we're racing at the moment
                    continue

                losses = np.ones((n_configurations_to_race, )) * INFINITY
                for j in range(n_configurations_to_race):
                    config_idx = configurations_to_race[j]
                    cutoff = cutoffs_to_race[j]
                    max_budget = config_to_budget_to_idx[config_idx][R_i[i]]

                    if time_elapsed[task_idx] > max_runtime:
                        losses[j] = 1.0
                        break

                    if not finites[j]:
                        runtime = np.nanmax(runtimes_memview[task_idx, config_idx])
                        if runtime > cutoff:
                            time_elapsed[task_idx] += cutoff
                        else:
                            time_elapsed[task_idx] += runtime
                        loss = 1.0
                        test_loss = 1.0
                        losses[j] = 1.0

                    else:

                        c = min(cutoff, max_runtime - time_elapsed[task_idx])

                        max_budget_for_algo = max_budget
                        for k in range(1, max_budget + 1):
                            if runtimes_memview[task_idx, config_idx, k] != runtimes_memview[task_idx, config_idx, k]:
                                max_budget_for_algo = k - 1
                                break

                        if c > runtimes_memview[task_idx, config_idx, max_budget_for_algo]:
                            loss = y_valid[task_idx, config_idx, max_budget_for_algo]
                            test_loss = y_test_memview[task_idx, config_idx, max_budget_for_algo]
                            time_elapsed[task_idx] += runtimes_memview[task_idx, config_idx, max_budget_for_algo]
                        elif runtimes_memview[task_idx, config_idx, 0] >= c:
                            time_elapsed[task_idx] += c
                            loss = 1.0
                            test_loss = 1.0
                        else:
                            loc = 0
                            for k in range(1, max_budget_for_algo + 1):
                                if c < runtimes_memview[task_idx, config_idx, k - 1]:
                                    break
                                else:
                                    loc = k
                            test_loss = y_test_memview[task_idx, config_idx, loc - 1]
                            time_elapsed[task_idx] += c
                            loss = y_valid[task_idx, config_idx, loc - 1]

                        losses[j] = loss

                    if losses[j] != losses[j]:
                        losses[j] = 1.0
                    if losses[j] == INFINITY:
                        raise ValueError(losses, j, loss, test_loss, task_idx, config_idx)

                    if loss < min_val_score[task_idx]:
                        min_val_score[task_idx] = loss
                        min_test_score[task_idx] = test_loss

                    if time_elapsed[task_idx] >= max_runtime:
                        break

                if i < s_max:
                    if N_i[i + 1] >= n_configurations_to_race:
                        print('This should not happen!')
                        # Do nothing if the next iteration expects more configurations than we're
                        # racing at the moment
                        raise ValueError()
                    else:
                        # When looking at the cython annotations this looks like
                        # it should be optimized away, but the code is so much
                        # more convenient like this...
                        argsort = losses.argsort(kind='mergesort')
                        configurations_to_race = configurations_to_race[argsort][:N_i[i + 1]]
                        cutoffs_to_race = cutoffs_to_race[argsort][:N_i[i + 1]]

        if min_test_score[task_idx] != min_test_score[task_idx] or min_test_score[task_idx] == INFINITY:
            min_test_score[task_idx] = 1.0

    # Cache only a full iteration of successive halving
    if len(configurations) % N_i[0] == 0:
        cache = dict()
        cache['time_elapsed'] = time_elapsed
        cache['min_val_score'] = min_val_score
        cache['min_test_score'] = min_test_score
        cache['skip_n_configs'] = len(configurations)

    return np.asarray(min_val_score), np.asarray(min_test_score), cache
