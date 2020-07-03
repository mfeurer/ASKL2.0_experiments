import logging
import typing

import numpy as np

from smac.intensification.successive_halving import SuccessiveHalving
from smac.optimizer.epm_configuration_chooser import EPMChooser
from smac.stats.stats import Stats
from smac.utils.constants import MAXINT
from smac.configspace import Configuration
from smac.runhistory.runhistory import RunHistory
from smac.tae.execute_ta_run import BudgetExhaustedException, CappedRunException, ExecuteTARun, StatusType
from smac.utils.io.traj_logging import TrajLogger


__author__ = "Ashwin Raaghav Narayanan"
__copyright__ = "Copyright 2019, ML4AAD"
__license__ = "3-clause BSD"


class OldSuccessiveHalving(SuccessiveHalving):

    def eval_challenger(self,
                        challenger: Configuration,
                        incumbent: typing.Optional[Configuration],
                        run_history: RunHistory,
                        time_bound: float = float(MAXINT),
                        log_traj: bool = True) -> typing.Tuple[Configuration, float]:
        """
        Running intensification via successive halving to determine the incumbent configuration.
        *Side effect:* adds runs to run_history

        Parameters
        ----------
        challenger : Configuration
            promising configuration
        incumbent : typing.Optional[Configuration]
            best configuration so far, None in 1st run
        run_history : smac.runhistory.runhistory.RunHistory
            stores all runs we ran so far
        time_bound : float, optional (default=2 ** 31 - 1)
            time in [sec] available to perform intensify
        log_traj : bool
            whether to log changes of incumbents in trajectory

        Returns
        -------
        typing.Tuple[Configuration, float]
            incumbent and incumbent cost
        """
        # calculating the incumbent's performance for adaptive capping
        # this check is required because:
        #   - there is no incumbent performance for the first ever 'intensify' run (from initial design)
        #   - during the 1st intensify run, the incumbent shouldn't be capped after being compared against itself
        if incumbent and incumbent != challenger:
            inc_runs = run_history.get_runs_for_config(incumbent, only_max_observed_budget=True)
            inc_sum_cost = run_history.sum_cost(config=incumbent, instance_seed_budget_keys=inc_runs)
        else:
            inc_sum_cost = np.inf
            if self.first_run:
                self.logger.info("First run, no incumbent provided; challenger is assumed to be the incumbent")
                incumbent = challenger
                self.first_run = False

        # select which instance to run current config on
        curr_budget = self.all_budgets[self.stage]

        # selecting instance-seed subset for this budget, depending on the kind of budget
        if self.instance_as_budget:
            prev_budget = int(self.all_budgets[self.stage - 1]) if self.stage > 0 else 0
            curr_insts = self.inst_seed_pairs[int(prev_budget):int(curr_budget)]
        else:
            curr_insts = self.inst_seed_pairs
        n_insts_remaining = len(curr_insts) - self.curr_inst_idx - 1

        self.logger.debug(" Running challenger  -  %s" % str(challenger))

        # run the next instance-seed pair for the given configuration
        instance, seed = curr_insts[self.curr_inst_idx]

        # selecting cutoff if running adaptive capping
        cutoff = self.cutoff
        if self.run_obj_time:
            cutoff = self._adapt_cutoff(challenger=challenger,
                                        run_history=run_history,
                                        inc_sum_cost=inc_sum_cost)
            if cutoff is not None and cutoff <= 0:
                # ran out of time to validate challenger
                self.logger.debug("Stop challenger intensification due to adaptive capping.")
                self.curr_inst_idx = np.inf

        self.logger.debug('Cutoff for challenger: %s' % str(cutoff))

        try:
            # run target algorithm for each instance-seed pair
            self.logger.debug("Execute target algorithm")

            try:
                status, cost, dur, res = self.tae_runner.start(
                    config=challenger,
                    instance=instance,
                    seed=seed,
                    cutoff=cutoff,
                    budget=0.0 if self.instance_as_budget else curr_budget,
                    instance_specific=self.instance_specifics.get(instance, "0"),
                    # Cutoff might be None if self.cutoff is None, but then the first if statement prevents
                    # evaluation of the second if statement
                    capped=(self.cutoff is not None) and (cutoff < self.cutoff)  # type: ignore[operator] # noqa F821
                )
                self._ta_time += dur
                self.num_run += 1
                self.curr_inst_idx += 1

            except CappedRunException:
                # We move on to the next configuration if a configuration is capped
                self.logger.debug("Budget exhausted by adaptive capping; "
                                  "Interrupting current challenger and moving on to the next one")
                # ignore all pending instances
                self.curr_inst_idx = np.inf
                n_insts_remaining = 0
                status = StatusType.CAPPED

            # adding challengers to the list of evaluated challengers
            #  - Stop: CAPPED/CRASHED/TIMEOUT/MEMOUT (!= SUCCESS & DONOTADVANCE)
            #  - Advance to next stage: SUCCESS
            # curr_challengers is a set, so "at least 1" success can be counted by set addition (no duplicates)
            # If a configuration is successful, it is added to curr_challengers.
            # if it fails it is added to fail_challengers.
            if np.isfinite(self.curr_inst_idx) and status in [StatusType.SUCCESS, StatusType.DONOTADVANCE]:
                self.success_challengers.add(challenger)  # successful configs
            else:
                self.fail_challengers.add(challenger)  # capped/crashed/do not advance configs

            # get incumbent if all instances have been evaluated
            if n_insts_remaining <= 0:
                incumbent = self._compare_configs(challenger=challenger,
                                                  incumbent=incumbent,
                                                  run_history=run_history,
                                                  log_traj=log_traj)
        except BudgetExhaustedException:
            # Returning the final incumbent selected so far because we ran out of optimization budget
            self.logger.debug("Budget exhausted; "
                              "Interrupting optimization run and returning current incumbent")

        # if all configurations for the current stage have been evaluated, reset stage
        num_chal_evaluated = len(self.success_challengers.union(self.fail_challengers)) + self.fail_chal_offset
        if num_chal_evaluated == self.n_configs_in_stage[self.stage] and n_insts_remaining <= 0:

            self.logger.info('Successive Halving iteration-step: %d-%d with '
                             'budget [%.2f / %d] - evaluated %d challenger(s)' %
                             (self.sh_iters + 1, self.stage + 1, self.all_budgets[self.stage], self.max_budget,
                              self.n_configs_in_stage[self.stage]))

            self._update_stage(run_history=run_history)

        # get incumbent cost
        inc_perf = run_history.get_cost(incumbent)

        return incumbent, inc_perf
