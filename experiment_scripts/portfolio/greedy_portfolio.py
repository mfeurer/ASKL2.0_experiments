from typing import Dict, Tuple, List, Optional, Any

import numpy as np
import pandas as pd

from fidelity_strategies import build_fidelity_strategy, FidelityStrategy
import portfolio_util


def _build_portfolio(
    y_test: np.ndarray,
    y_valid: Optional[np.ndarray],
    runtimes_matrix: np.ndarray,
    config_ids: List[int],
    config_id_to_idx: Dict[int, int],
    config_to_budget_to_idx: Dict[str, Dict[float, int]],
    task_id_to_idx: Dict[int, int],
    portfolio_size: int,
    rng: np.random.RandomState,
    losses: Optional[Dict[int, float]],
    fidelity_strategy: FidelityStrategy,
) -> Tuple[List[str], np.ndarray, List[Dict[float, int]]]:
    shuffled_config_ids = rng.permutation(list(config_ids))
    if y_valid is None:
        y_valid = y_test

    portfolio = []
    budget_to_idx = []
    cache_2 = None

    old_performances = np.ones((len(y_test),))
    if losses:
        for task_id in losses:
            task_idx = task_id_to_idx[task_id]
            old_performances[task_idx] = losses[task_id]

    for i in range(portfolio_size):
        scores = []
        caches_2 = []

        # Define these here to have the code more similar to
        # _build_portfolio_with_cutoff
        cutoffs = np.ones((i + 1,)) * np.inf
        runtimes = runtimes_matrix.copy()
        runtimes[np.isfinite(runtimes)] = 0.0
        max_runtime = np.inf

        for j, config_id in enumerate(shuffled_config_ids):
            if config_id in portfolio:
                scores.append(np.inf)
                caches_2.append(None)
            else:

                portfolio.append(config_id)

                # # for-loop-based version, written in pure Python. Comment this in to check
                # # the vectorized version in Cython, which is way more involved
                # scores_j = []
                # for idx in range(len(y_valid)):
                #     score = fidelity_strategy.play(
                #         y_valid=y_valid[idx],
                #         y_test=y_test[idx],
                #         runtimes=runtimes[idx],
                #         configurations=portfolio,
                #         config_id_to_idx=config_id_to_idx,
                #         config_to_budget_to_idx=config_to_budget_to_idx,
                #         cutoffs=cutoffs,
                #         max_runtime=max_runtime,
                #     )[1]
                #     scores_j.append(score)

                # cython + vectorized
                _, test_wise_scores_2, cache_j2 = \
                    fidelity_strategy.play_cythonized_vectorized(
                        y_valid=y_valid,
                        y_test=y_test,
                        runtimes=runtimes,
                        configurations=portfolio,
                        config_id_to_idx=config_id_to_idx,
                        cutoffs=cutoffs,
                        config_to_budget_to_idx=config_to_budget_to_idx,
                        max_runtime=float(max_runtime),
                        cache=cache_2,
                    )

                # np.testing.assert_array_almost_equal(
                #     scores_j, test_wise_scores_2,
                # )

                caches_2.append(cache_j2)

                del portfolio[-1]
                # One can interchange scores_j and test_wise_scores here
                test_wise_scores = np.minimum(old_performances, test_wise_scores_2)
                scores.append(np.mean(test_wise_scores))

        argmin = int(np.argmin(scores))  # type: int
        config_id = shuffled_config_ids[argmin]
        print(i, scores[argmin], config_id)
        portfolio.append(config_id)
        budget_to_idx.append(config_to_budget_to_idx[config_id_to_idx[config_id]])
        if len(caches_2) == len(scores):
            cache_2 = caches_2[argmin]

        # If converged!
        if np.min(scores) <= 0:
            break

    return portfolio, np.array([np.inf] * len(portfolio), dtype=np.float64), budget_to_idx


def _build_portfolio_with_cutoffs(
    y_test: np.ndarray,
    y_valid: Optional[np.ndarray],
    runtimes_matrix: np.ndarray,
    config_ids: List[int],
    config_id_to_idx: Dict[int, int],
    config_to_budget_to_idx: Dict[str, Dict[float, int]],
    task_id_to_idx: Dict[int, int],
    portfolio_size: int,
    rng: np.random.RandomState,
    max_runtime: int,
    losses: Optional[Dict[int, float]],
    fidelity_strategy: FidelityStrategy,
) -> Tuple[List[str], np.ndarray, List[Dict[float, int]]]:
    shuffled_config_ids = rng.permutation(list(config_ids))
    if y_valid is None:
        y_valid = y_test

    old_performances = np.ones((len(y_test),))
    if losses:
        for task_id in losses:
            task_idx = task_id_to_idx[task_id]
            old_performances[task_idx] = losses[task_id]

    nanmax = np.nanmax(runtimes_matrix) + 1
    factor = 2
    ts = ([
        int(2 ** (exponent / factor)) for exponent in
        range(0, factor * int(np.ceil(np.log2(max_runtime))))
        if 2 ** (exponent / factor) <= (max_runtime / 2)
    ] + [max_runtime / 2, max_runtime])
    if nanmax > 0 and nanmax < max_runtime:
        ts.append(nanmax)
    if portfolio_size == 1:
        ts += [max_runtime]
    ts = np.unique(ts)

    scores_t = []
    portfolios_t = []
    budget_to_idx_t = []
    cutoffs_t = []
    n_iter_above_max_observed_runtime = 0

    for t_idx, t in enumerate(ts):

        cache_2 = None

        if (
            t_idx + 1 < len(ts)
            and (ts[t_idx] * portfolio_size < max_runtime)
            and ((ts[t_idx + 1]) * portfolio_size < max_runtime)
        ):
            print('Skipping cutoff', t)
            continue

        if n_iter_above_max_observed_runtime > 0:
            print('Skipping cutoff', t)
            continue
        if t > np.nanmax(runtimes_matrix):
            print(
                'Cutoff %f larger than nanmax %f of runtimes matrix' %
                (t, float(np.nanmax(runtimes_matrix)))
            )
            n_iter_above_max_observed_runtime += 1

        portfolio = []
        budget_to_idx = []
        trajectory = []
        cutoffs = None

        for i in range(portfolio_size):
            scores = []
            caches_2 = []
            cutoffs = np.array([t] * (i + 1), dtype=np.float64)

            for j, config_id in enumerate(shuffled_config_ids):

                if config_id in portfolio:
                    scores.append(np.inf)
                    caches_2.append(None)

                else:

                    portfolio.append(config_id)

                    # # For-loop based version
                    # scores_per_task = []
                    # for idx in range(len(y_valid)):
                    #     score = fidelity_strategy.play(
                    #         y_valid=y_valid[idx],
                    #         y_test=y_test[idx],
                    #         runtimes=runtimes_matrix[idx],
                    #         configurations=portfolio,
                    #         config_id_to_idx=config_id_to_idx,
                    #         config_to_budget_to_idx=config_to_budget_to_idx,
                    #         cutoffs=cutoffs,
                    #         max_runtime=np.float64(max_runtime),
                    #     )[1]
                    #     scores_per_task.append(score)

                    # Cythonized version
                    _, test_wise_scores_2, cache_j2 = (
                        fidelity_strategy.play_cythonized_vectorized(
                            y_valid=y_valid,
                            y_test=y_test,
                            runtimes=runtimes_matrix,
                            configurations=portfolio,
                            config_id_to_idx=config_id_to_idx,
                            config_to_budget_to_idx=config_to_budget_to_idx,
                            cutoffs=cutoffs,
                            max_runtime=np.float64(max_runtime),
                            cache=cache_2,
                        )
                    )

                    # scores_per_task = np.array(scores_per_task, dtype=test_wise_scores_2.dtype)
                    # try:
                    #     np.testing.assert_array_almost_equal(
                    #         scores_per_task, test_wise_scores_2,
                    #         err_msg=str(
                    #             (
                    #                 t,
                    #                 portfolio,
                    #                 list(scores_per_task),
                    #                 list(test_wise_scores_2),
                    #             )
                    #         )
                    #     )
                    # except AssertionError:
                    #     print(scores_per_task.dtype, test_wise_scores_2.dtype)
                    #     for s1, s2 in zip(scores_per_task, test_wise_scores_2):
                    #         print(s1, s2, type(s1), type(s2), abs(s1 - s1))
                    #     raise

                    # One can interchange scores_j and test_wise_scores here
                    test_wise_scores = np.minimum(old_performances, test_wise_scores_2)
                    # print(j, t, np.mean(test_wise_scores), test_wise_scores)

                    del portfolio[-1]
                    scores.append(test_wise_scores.mean())
                    caches_2.append(cache_j2)

            argmin = int(np.argmin(scores))  # type: int
            trajectory.append(scores[argmin])
            config_id = shuffled_config_ids[argmin]
            print(i, scores[argmin], cutoffs[0])

            portfolio.append(config_id)
            budget_to_idx.append(config_to_budget_to_idx[config_id_to_idx[config_id]])
            if len(caches_2) == len(scores):
                cache_2 = caches_2[argmin]

            # If converged!
            if np.min(scores) <= 0:
                break

        scores_t.append(trajectory[-1])
        portfolios_t.append(portfolio)
        budget_to_idx_t.append(budget_to_idx)
        if cutoffs is None:
            raise ValueError('Cutoffs array should not be None!')
        cutoffs_t.append(cutoffs)

    argmin = int(np.argmin(scores_t))
    print('Selecting cutoff %f, score %f' % (cutoffs_t[argmin][0], scores_t[argmin]))
    print({cutoffs_t[i][0]: scores_t[i] for i in range(len(cutoffs_t))})
    portfolio = portfolios_t[argmin]
    budget_to_idx = budget_to_idx_t[argmin]
    cutoffs = np.array(cutoffs_t[argmin], dtype=np.float64)

    print(portfolio, cutoffs)
    return portfolio, cutoffs, budget_to_idx


def build(
    matrix: Dict[int, pd.DataFrame],
    task_ids: List[int],
    configurations: Dict[str, Dict],
    portfolio_size: int,
    max_runtime: int,
    rng: np.random.RandomState,
    fidelity_strategy_name: str,
    fidelity_strategy_kwargs: Dict[str, Any],
    losses: Optional[Dict[int, float]] = None,
    consider_validation: bool = False,
) -> Tuple[List[str], np.ndarray, List[Dict[float, int]]]:
    y_valid, y_test, runtimes, config_id_to_idx, task_id_to_idx = portfolio_util.reformat_data(
        matrix, task_ids, configurations)
    print('Building portfolio on matrix of shape %s' % str(y_test.shape))
    normalized_matrix = portfolio_util.normalize_matrix(y_test)

    fidelity_strategy = build_fidelity_strategy(
        fidelity_strategy_name=fidelity_strategy_name,
        kwargs=fidelity_strategy_kwargs,
    )
    print('Using fidelity strategy', fidelity_strategy)

    config_to_budget_to_idx = {}
    for config_id, config in configurations.items():
        config_idx = config_id_to_idx[config_id]
        if config['classifier:__choice__'] in ['sgd', 'passive_aggressive']:
            budget_to_idx = {
                0.1953125: 0,
                0.390625: 1,
                0.78125: 2,
                1.5625: 3,
                3.125: 4,
                6.25: 5,
                12.5: 6,
                25: 7,
                50: 8,
                100: 9
            }
        else:
            budget_to_idx = {
                0.390625: 0,
                0.78125: 1,
                1.5625: 2,
                3.125: 3,
                6.25: 4,
                12.5: 5,
                25: 6,
                50: 7,
                100: 8
            }
        config_to_budget_to_idx[config_idx] = budget_to_idx

    if max_runtime is None or not np.isfinite(max_runtime):
        kwargs = dict(
            y_test=normalized_matrix,
            runtimes_matrix=runtimes,
            config_ids=list(configurations),
            config_id_to_idx=config_id_to_idx,
            config_to_budget_to_idx=config_to_budget_to_idx,
            task_id_to_idx=task_id_to_idx,
            portfolio_size=portfolio_size,
            rng=rng,
            losses=losses,
            fidelity_strategy=fidelity_strategy,
        )
        if consider_validation:
            portfolio, cutoffs, budget_to_idx = _build_portfolio(
                y_valid=y_valid.copy(),
                **kwargs
            )
        else:
            portfolio, cutoffs, budget_to_idx = _build_portfolio(
                y_valid=None,
                **kwargs
            )
    else:
        kwargs = dict(
            y_test=normalized_matrix,
            runtimes_matrix=runtimes,
            config_ids=list(configurations),
            config_id_to_idx=config_id_to_idx,
            config_to_budget_to_idx=config_to_budget_to_idx,
            task_id_to_idx=task_id_to_idx,
            portfolio_size=portfolio_size,
            rng=rng,
            max_runtime=max_runtime,
            losses=losses,
            fidelity_strategy=fidelity_strategy,
        )
        if consider_validation:
            portfolio, cutoffs, budget_to_idx = _build_portfolio_with_cutoffs(
                y_valid=y_valid.copy(),
                **kwargs
            )
        else:
            portfolio, cutoffs, budget_to_idx = _build_portfolio_with_cutoffs(
                y_valid=None,
                **kwargs
            )

    return portfolio, cutoffs, budget_to_idx

