import argparse
import glob
import json
import logging
import numpy as np
import os
import tempfile
import time
import warnings

import sys

from scipy.sparse import coo_matrix
from sklearn.utils.validation import check_random_state
from sklearn.utils.multiclass import unique_labels

sys.path.append('..')
from utils import load_task

from autosklearn.ensemble_builder import EnsembleBuilder
from autosklearn.constants import BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION
from autosklearn.metrics import balanced_accuracy, make_scorer
from autosklearn.util.backend import BackendContext, Backend


class BalancedAccuracy:

    def __init__(self):
        self.sample_weight = None

    def confusion_matrix(self, y_true, y_pred, labels=None, sample_weight=None,
                         normalize=None):

        if labels is None:
            labels = unique_labels(y_true, y_pred)
        else:
            labels = np.asarray(labels)
            if np.all([l not in y_true for l in labels]):
                raise ValueError("At least one label specified must be in y_true")

        if self.sample_weight is None:
            self.sample_weight = np.ones(y_true.shape[0], dtype=np.int64)

        n_labels = labels.size

        # # intersect y_pred, y_true with labels, eliminate items not in labels
        ind = np.logical_and(y_pred < n_labels, y_true < n_labels)
        y_pred = y_pred[ind]
        y_true = y_true[ind]
        # also eliminate weights of eliminated items
        sample_weight = self.sample_weight[ind]

        cm = coo_matrix((sample_weight, (y_true, y_pred)),
                        shape=(n_labels, n_labels), dtype=np.int64,
                        ).toarray()

        with np.errstate(all='ignore'):
            if normalize == 'true':
                cm = cm / cm.sum(axis=1, keepdims=True)
            elif normalize == 'pred':
                cm = cm / cm.sum(axis=0, keepdims=True)
            elif normalize == 'all':
                cm = cm / cm.sum()
            cm = np.nan_to_num(cm)

        return cm

    def __call__(self, y_true, y_pred, sample_weight=None, adjusted=False):
        y_true = y_true.astype(np.int64)
        C = self.confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
        with np.errstate(divide='ignore', invalid='ignore'):
            per_class = np.diag(C) / C.sum(axis=1)
        if np.any(np.isnan(per_class)):
            warnings.warn('y_pred contains classes not in y_true')
            per_class = per_class[~np.isnan(per_class)]
        score = np.mean(per_class)
        if adjusted:
            n_classes = len(per_class)
            chance = 1 / n_classes
            score -= chance
            score /= 1 - chance
        return score


# BackendContext
class BackendContextMock(BackendContext):

    # Do not create directories
    def create_directories(self):
        if os.path.isdir(self.temporary_directory):
            self._tmp_dir_created = True

        if os.path.isdir(self.output_directory):
            self._output_dir_created = True


def main(task_id, ensemble_dir, performance_range_threshold, ensemble_size, max_keep_best, seed,
         only_portfolio_runs, call_from_cmd):

    if max_keep_best > 1:
        assert max_keep_best == int(max_keep_best)
        max_keep_best = int(max_keep_best)

    memory_limit = 4000
    precision = 32
    metric = make_scorer('balanced_accuracy_fast', BalancedAccuracy())

    if not os.path.exists(ensemble_dir):
        raise NotADirectoryError("%s does not exist")
    if call_from_cmd:
        assert str(task_id) in ensemble_dir

    fl_name = "ensemble_results_%fthresh_%dsize_%fbest" % \
              (performance_range_threshold, ensemble_size, max_keep_best)
    if only_portfolio_runs:
        fl_name += "_only_portfolio"
    fl_name = os.path.join(ensemble_dir, fl_name)
    if os.path.isfile(fl_name):
        raise ValueError("Nothing left to do, %s already exists" % fl_name)

    # figure out how many prediction files are in dir
    if call_from_cmd:
        pred_dir = os.path.join(ensemble_dir, "auto-sklearn-output", ".auto-sklearn", "predictions_ensemble")
        n_models = glob.glob(pred_dir + "/predictions_ensemble_%d_*.npy.gz" % seed)
    else:
        pred_dir = os.path.join(ensemble_dir, ".auto-sklearn", "predictions_ensemble")
        n_models = glob.glob(pred_dir + "/predictions_ensemble_%d_*.npy" % seed)
    n_models.sort(key=lambda x: int(float(x.split("_")[-2])))
    print("\n".join(n_models))
    print("Found %d ensemble predictions" % len(n_models))
    if len(n_models) == 0:
        raise ValueError("%s has no ensemble predictions" % pred_dir)

    # Get start time of ensemble building: 1) load json 2) find key 3) get creation times
    if call_from_cmd:
        timestamps_fl = os.path.join(ensemble_dir, "auto-sklearn-output", "timestamps.json")
    else:
        timestamps_fl = os.path.join(ensemble_dir, "timestamps.json")
    with open(timestamps_fl, "r") as fh:
        timestamps = json.load(fh)
    model_timestamps = None
    overall_start_time = None
    for k in timestamps:
        if "predictions_ensemble" in k:
            model_timestamps = timestamps[k]
        if "start_time_%d" % seed in timestamps[k]:
            overall_start_time = timestamps[k]["start_time_%d" % seed]
    timestamp_keys = list(model_timestamps.keys())
    for timestamp_key in timestamp_keys:
        if timestamp_key.endswith('lock') or 'predictions_ensemble' not in timestamp_key:
            del model_timestamps[timestamp_key]
    assert model_timestamps is not None and overall_start_time is not None
    assert len(model_timestamps) == len(n_models), (len(model_timestamps), len(n_models))
    # Get overall timelimit
    vanilla_results_fl = os.path.join(ensemble_dir, "result.json")
    with open(vanilla_results_fl, "r") as fh:
        vanilla_results = json.load(fh)

    # If only portfolio configurations, read runhistory
    if only_portfolio_runs:
        if call_from_cmd:
            runhistory_fl = os.path.join(ensemble_dir, "auto-sklearn-output", "smac3-output",
                                         "run*", "runhistory.json")
        else:
            runhistory_fl = os.path.join(ensemble_dir, "smac3-output",
                                         "run*", "runhistory.json")
        runhistory_fl = glob.glob(runhistory_fl)
        assert len(runhistory_fl) == 1
        with open(runhistory_fl[0], "r") as fh:
            runhistory = json.load(fh)

        init_design_num_runs = []
        for i in runhistory["data"]:
            if i[1][3]["configuration_origin"] == "Initial design":
                if "error" in i[1][3]:
                    continue
                init_design_num_runs.append(i[1][3]["num_run"])
        print("Portfolio stopped after %s runs" % str(init_design_num_runs))
        last_run = max(init_design_num_runs)
        print("Cut down to only portfolio runs fom %d" % len(n_models))
        for i, n in enumerate(n_models):
            if int(float(n.split("_")[-2])) > last_run:
                n_models = n_models[:i]
                break
        print("... to %d" % len(n_models))

    # load data
    X_train, y_train, X_test, y_test, cat = load_task(task_id)

    if len(np.unique(y_test)) == 2:
        task_type = BINARY_CLASSIFICATION
    elif len(np.unique(y_test)) > 2:
        task_type = MULTICLASS_CLASSIFICATION
    else:
        raise ValueError("Unknown task type for task %d" % task_id)

    tmp_dir = tempfile.TemporaryDirectory()
    loss_trajectory = []

    # Construct ensemble builder
    context = BackendContextMock(
        temporary_directory=(ensemble_dir + "/auto-sklearn-output/" if call_from_cmd else ensemble_dir),
        output_directory=tmp_dir.name,
        delete_tmp_folder_after_terminate=False,
        delete_output_folder_after_terminate=False,
        shared_mode=False
    )
    backend = Backend(context)

    ens_builder = EnsembleBuilder(
        backend=backend,
        dataset_name=str(task_id),
        task_type=task_type,
        metric=metric,
        limit=np.inf,
        ensemble_size=ensemble_size,
        ensemble_nbest=max_keep_best,
        performance_range_threshold=performance_range_threshold,
        max_models_on_disc=None,
        seed=seed,
        shared_mode=False,
        precision=precision,
        max_iterations=1,
        read_at_most=1,
        memory_limit=memory_limit,
        random_state=1,
        sleep_duration=0
    )

    try:
        # iterate over all models, take construction time into account when creating new trajectory
        current_ensemble_timestamp = 0
        skipped = 1
        for midx, model_path in enumerate(n_models):
            tstamp = model_timestamps[
                model_path.split("/")[-1].replace('.gz', '')] - overall_start_time
            if current_ensemble_timestamp > tstamp:
                # while this model was built, the ensemble script was not yet done
                skipped += 1
                continue

            # Do one ensemble building step
            start = time.time()
            ens_builder.random_state = check_random_state(1)
            print("############## %d: Working on %s (skipped %d)" % (midx+1, model_path, skipped-1))
            logging.basicConfig(level=logging.DEBUG)
            ens_builder.read_at_most = skipped
            valid_pred, test_pred = ens_builder.main(return_pred=True)
            last_dur = time.time() - start
            current_ensemble_timestamp = tstamp + last_dur

            if current_ensemble_timestamp >= vanilla_results["0"]["time_limit"]:
                print("############## Went over time %f > %f; Stop here" %
                      (current_ensemble_timestamp, vanilla_results["0"]["time_limit"]))
                break

            # Reset, since we have just read model files
            skipped = 1
            if test_pred is None:
                # Adding this model did not change the ensemble, no new prediction
                continue
            if task_type == BINARY_CLASSIFICATION:
                # Recreate nx2 array
                test_pred = np.concatenate([1-test_pred.reshape([-1, 1]), test_pred.reshape([-1, 1])], axis=1)

            # Build trajectory entry
            score = 1 - balanced_accuracy(y_true=y_test, y_pred=test_pred)
            loss_trajectory.append((current_ensemble_timestamp, score))
            print("############## Round %d took %g sec" % (midx, time.time() - start))
    except:
        raise
    finally:
        tmp_dir.cleanup()

    # Store results
    result = dict()
    result[ensemble_size] = {
        'task_id': task_id,
        'time_limit': vanilla_results["0"]["time_limit"],
        'loss': loss_trajectory[-1][1],
        'configuration': {
            "n_models": n_models,
            "performance_range_threshold": performance_range_threshold,
            "ensemble_size": ensemble_size,
            "max_keep_best": max_keep_best,
            "seed": seed,
            "memory_limit": memory_limit,
            "precision": precision,
            },
        'n_models': len(n_models),
        'trajectory': loss_trajectory,
    }

    with open(fl_name, 'wt') as fh:
        json.dump(result, fh, indent=4)
    print("Dumped to %s" % fl_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ensemble-dir', type=str, required=True)
    parser.add_argument('--task-id', type=int, required=True)
    parser.add_argument('--max-keep-best', type=float, required=True)
    parser.add_argument('--ensemble-size', type=int, required=True)
    parser.add_argument('--performance-range-threshold', type=float, required=True)
    parser.add_argument('--only-portfolio-runs', default=False, action="store_true")
    parser.add_argument('--seed', type=int, required=True)

    args = parser.parse_args()

    task_id = args.task_id
    ensemble_dir = args.ensemble_dir

    performance_range_threshold = args.performance_range_threshold
    ensemble_size = args.ensemble_size
    max_keep_best = args.max_keep_best
    seed = args.seed
    only_portfolio_runs = args.only_portfolio_runs
    main(task_id=task_id, ensemble_dir=ensemble_dir,
         performance_range_threshold=performance_range_threshold,
         ensemble_size=ensemble_size, max_keep_best=max_keep_best, seed=seed,
         only_portfolio_runs=only_portfolio_runs, call_from_cmd=True)
