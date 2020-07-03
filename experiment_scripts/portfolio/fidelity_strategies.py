from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd

import pyx.no_fidelities
import pyx.sh


class FidelityStrategy(object):

    def __init__(self, name):
        self.name = name

    def play(
        self,
        y_valid: np.ndarray,
        y_test: np.ndarray,
        runtimes: np.ndarray,
        configurations: List[str],
        config_id_to_idx: Dict[int, str],
        cutoffs: np.ndarray,
        max_runtime: float,
        config_to_budget_to_idx: Dict[str, Dict[float, int]],
    ) -> Tuple[float, float, pd.Series]:

        raise NotImplementedError()

    def play_cythonized_vectorized(
            self,
            y_valid: np.ndarray,
            y_test: np.ndarray,
            runtimes: np.ndarray,
            configurations: List[str],
            config_id_to_idx: Dict[int, str],
            cutoffs: np.ndarray,
            max_runtime: float,
            config_to_budget_to_idx: Dict[str, Dict[float, int]],
            cache: Optional[Dict] = None,
            n_jobs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        raise NotImplementedError


class HoldoutIterativeFit(FidelityStrategy):

    def __init__(self):
        super().__init__('holdout-iterative-fit')

    def play(
        self,
        y_valid: np.ndarray,
        y_test: np.ndarray,
        runtimes: np.ndarray,
        configurations: List[str],
        config_id_to_idx: Dict[int, str],
        cutoffs: np.ndarray,
        max_runtime: float,
        config_to_budget_to_idx: Dict[str, Dict[float, int]],
    ) -> Tuple[np.ndarray, np.ndarray, pd.Series]:
        assert len(y_valid.shape) == 2, y_valid.shape
        assert len(y_test.shape) == 2, y_test.shape
        assert len(runtimes.shape) == 2, runtimes.shape

        time_elapsed = 0
        min_val_score = np.inf
        min_test_score = np.inf
        trajectory = {time_elapsed: 1}

        # Going through all portfolio members
        for member, cutoff in zip(configurations, cutoffs):
            config_idx = config_id_to_idx[member]

            if time_elapsed >= max_runtime:
                break

            # Check if there are no successful entries for this configuration
            if not np.any(np.isfinite(y_valid[config_idx])):
                runtime = np.nanmax(runtimes[config_idx])
                if runtime > cutoff:
                    time_elapsed += cutoff
                else:
                    time_elapsed += runtime
                loss = 1.0
                test_loss = 1.0

            # If there are successful entries for this configuration
            else:
                lc = y_valid[config_idx]
                keep = np.isfinite(lc)
                lc = lc[keep]
                lc_test = y_test[config_idx][keep]
                lc_runtime = runtimes[config_idx][keep]

                # Clamp the cutoff to not run over the total max_runtime
                c = min(cutoff, max_runtime - time_elapsed)

                # Reading out the performance. Depending on the runtime of the algorithm and the
                # cutoff, we end up in one of three different states:
                # 1. The cutoff is higher than the total runtime of the algorithm. We can read out
                #    the final performance value.
                if c > lc_runtime[-1]:
                    loss = lc[-1]
                    test_loss = lc_test[-1]
                    time_elapsed += lc_runtime[-1]
                # 2. The cutoff is higher than the runtime for training the algorithm for a single
                #    iteration. This is a clear failure.
                elif c < lc_runtime[0]:
                    time_elapsed += c
                    loss = 1.0
                    test_loss = 1.0
                # 3. We were able to train the algorithm for a few steps, but not for the full
                #    amount of steps because we ran out of time before. We need to find the point
                #    where an evaluation was still within the time limit.
                else:
                    # Ignore the PEP8 issues here, the == compares element
                    # wise with the numpy array
                    loc = len(lc_runtime[(lc_runtime < c) == True])
                    lc = lc[:loc]
                    lc_test = lc_test[:loc]
                    loss = lc[-1]
                    test_loss = lc_test[-1]
                    time_elapsed += c

            # Update the best loss.
            if loss < min_val_score:
                min_val_score = loss
                min_test_score = test_loss
                trajectory[time_elapsed] = test_loss

            if time_elapsed >= max_runtime:
                break

        if not np.isfinite(min_test_score):
            min_test_score = 1.0

        return min_val_score, min_test_score, pd.Series(trajectory)

    def play_cythonized_vectorized(
        self,
        y_valid: np.ndarray,
        y_test: np.ndarray,
        runtimes: np.ndarray,
        configurations: List[str],
        config_id_to_idx: Dict[int, str],
        cutoffs: np.ndarray,
        max_runtime: float,
        config_to_budget_to_idx: Dict[str, Dict[float, int]],
        cache: Optional[Dict] = None,
        n_jobs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        return pyx.no_fidelities.play_vectorized(
            y_valid=y_valid,
            y_test=y_test,
            runtimes=runtimes,
            configurations=configurations,
            config_id_to_idx=config_id_to_idx,
            cutoffs=cutoffs,
            max_runtime=max_runtime,
            cache=cache,
        )


class RepeatedSuccessiveHalving(FidelityStrategy):

    def __init__(self, eta, min_budget, max_budget):
        super().__init__('sh')
        self.eta = eta
        self.min_budget = min_budget
        self.max_budget = max_budget

        # https://openreview.net/pdf?id=ry18Ww5ee
        self.R = self.max_budget / self.min_budget
        self.s_max = int(np.floor(np.log(self.R) / np.log(self.eta)))
        self.B = (self.s_max + 1) * self.R
        n = int(np.ceil((self.B / self.R) * (np.power(self.eta, self.s_max) / (self.s_max + 1))))

        r = self.R * np.power(float(self.eta), -self.s_max)
        if np.rint(self.R) != self.R:
            raise ValueError('R is not an integer, but %f' % self.R)

        # Number of configurations to keep in this iteration
        self.N_i = np.array([
            int(np.floor(n * np.power(float(self.eta), -i)))
            for i in range(self.s_max + 1)
        ], dtype=np.int32)
        # Budget to assign each configuration, this is different to the paper
        # in that it multiplies by the min_budget. necessary to accomatade for
        # the # calculation of R.
        self.R_i = np.array([
            r * np.power(self.eta, i) * self.min_budget
            for i in range(self.s_max + 1)
        ], dtype=np.float64)

    def play(
        self,
        y_valid: np.ndarray,
        y_test: np.ndarray,
        runtimes: np.ndarray,
        configurations: List[str],
        config_id_to_idx: Dict[int, str],
        cutoffs: np.ndarray,
        max_runtime: float,
        config_to_budget_to_idx: Dict[str, Dict[float, int]],
    ) -> Tuple[np.ndarray, np.ndarray, pd.Series]:
        # TODO: one could add a simple pruning strategy: if the last configuration of
        # the portfolio is not advanced to the next budget we could terminate running
        # this successive halving run

        time_elapsed = 0
        min_val_score = np.inf
        min_test_score = np.inf
        trajectory = {time_elapsed: 1}

        while len(configurations) > 0:
            n_i = self.N_i[0]
            configurations_to_race = configurations[:n_i]
            configurations = configurations[n_i:]
            cutoffs_to_race = cutoffs[:n_i]
            cutoffs = cutoffs[n_i:]

            # This here is one iteration of successive halving!
            for i in range(self.s_max + 1):

                if i < self.s_max and self.N_i[i + 1] >= len(configurations_to_race):
                    # Do nothing if the next iteration expects more
                    # configurations than we're racing at the moment
                    continue

                losses = np.zeros((len(configurations_to_race),))
                for j, (member, cutoff) in enumerate(zip(configurations_to_race, cutoffs_to_race)):

                    config_idx = config_id_to_idx[member]
                    max_budget = config_to_budget_to_idx[config_idx][self.R_i[i]]

                    if time_elapsed > max_runtime:
                        losses[j] = 1.0
                        break

                    # These lines might be optimized away (because they're
                    # called multiple times for each configuration),
                    # but let's keep them for simplicity
                    if not np.any(np.isfinite(y_valid[config_idx])):
                        runtime = np.nanmax(runtimes[config_idx])
                        if runtime > cutoff:
                            time_elapsed += cutoff
                        else:
                            time_elapsed += runtime
                        loss = 1.0
                        test_loss = 1.0
                        losses[j] = 1.0

                    else:

                        lc = y_valid[config_idx, :max_budget + 1]
                        keep = np.isfinite(lc)
                        lc = lc[keep]
                        lc_test = y_test[config_idx, :max_budget + 1][keep]
                        lc_runtime = runtimes[config_idx, :max_budget + 1][keep]

                        c = min(cutoff, max_runtime - time_elapsed)
                        if c > lc_runtime[-1]:
                            loss = lc[-1]
                            test_loss = lc_test[-1]
                            time_elapsed += lc_runtime[-1]
                        elif c < lc_runtime[0]:
                            time_elapsed += c
                            loss = 1.0
                            test_loss = 1.0
                        else:
                            loc = len(lc_runtime[(lc_runtime < c) == True])
                            lc = lc[:loc]
                            lc_test = lc_test[:loc]
                            loss = lc[-1]
                            test_loss = lc_test[-1]
                            time_elapsed += c

                        losses[j] = loss

                    if np.isnan(losses[j]):
                        losses[j] = 1.0
                    if np.isinf(losses[j]):
                        raise ValueError(losses, j, loss, test_loss, config_idx)
                    if loss < min_val_score:
                        min_val_score = loss
                        min_test_score = test_loss
                        trajectory[time_elapsed] = test_loss

                    if time_elapsed >= max_runtime:
                        break

                if i < self.s_max:
                    if self.N_i[i + 1] >= len(configurations_to_race):
                        print('This should not happen!')
                        # Do nothing if the next iteration expects more configurations than we're
                        # racing at the moment
                        raise ValueError()
                    else:
                        argsort = np.argsort(losses, kind='mergesort')
                        configurations_to_race = [configurations_to_race[arg] for arg in argsort][:self.N_i[i + 1]]
                        cutoffs_to_race = [cutoffs_to_race[arg] for arg in argsort][:self.N_i[i + 1]]

        if not np.isfinite(min_test_score):
            min_test_score = 1.0

        return min_val_score, min_test_score, pd.Series(trajectory)

    def play_cythonized_vectorized(
        self,
        y_valid: np.ndarray,
        y_test: np.ndarray,
        runtimes: np.ndarray,
        configurations: List[str],
        config_id_to_idx: Dict[int, str],
        cutoffs: np.ndarray,
        max_runtime: float,
        config_to_budget_to_idx: Dict[str, Dict[float, int]],
        cache: Optional[Dict] = None,
        n_jobs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        return pyx.sh.play_vectorized(
            y_valid=y_valid,
            y_test=y_test,
            runtimes=runtimes,
            configurations=configurations,
            config_id_to_idx=config_id_to_idx,
            cutoffs=cutoffs,
            max_runtime=max_runtime,
            config_to_budget_to_idx=config_to_budget_to_idx,
            N_i=self.N_i,
            s_max=self.s_max,
            R_i=self.R_i,
            cache=cache,
        )


def build_fidelity_strategy(
    fidelity_strategy_name: str,
    kwargs: Dict[str, Any],
) -> FidelityStrategy:
    if fidelity_strategy_name.lower() in ['none', 'nofidelity']:
        if len(kwargs) > 0:
            raise ValueError(kwargs)
        return HoldoutIterativeFit()
    elif fidelity_strategy_name.lower() == 'sh':
        return RepeatedSuccessiveHalving(**kwargs)
    else:
        raise ValueError(fidelity_strategy_name)
