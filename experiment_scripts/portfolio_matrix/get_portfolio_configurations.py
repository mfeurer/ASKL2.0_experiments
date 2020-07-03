import argparse
from collections import defaultdict
import glob
import hashlib
import json
import os
import pickle
import sys

from ConfigSpace.read_and_write.json import read, write
from ConfigSpace import Configuration
import numpy as np
import openml
import pandas as pd
from smac.runhistory.runhistory import RunHistory, StatusType
from sklearn.externals.joblib import Parallel, delayed

sys.path.append("..")
from utils import automl_metadata, dataset_dc, method_dc, openml_automl_benchmark


def read_configurations_for_task_id(task_id, task_id_to_dir, cs):
    incumbents_test_rval = list()

    rh = RunHistory()
    for entry in task_id_to_dir[task_id]:
        # Merge all evaluations from multiple SMAC runs into one runhistory
        rh.update_from_json(entry, cs)

    X = []
    run_times = []
    Y_train = []
    Y_test = []
    status = []
    results = {key.config_id: value for key, value in rh.data.items()}
    max_lc_length = 0

    for config_id in results:
        run_times_tmp = []
        y_train = []
        y_test = []

        if results[config_id].status == StatusType.SUCCESS:
            run_times_tmp.append(results[config_id].time)
            y_train.append(results[config_id].additional_info['train_loss'])
            y_test.append(results[config_id].additional_info['test_loss'])
            status.append(0)
        else:
            run_times_tmp.append(results[config_id].time)
            y_train.append(1.0)
            y_test.append(1.0)
            status.append(1)

        X.append(rh.ids_config[config_id])
        run_times.append(run_times_tmp)
        Y_train.append(y_train)
        Y_test.append(y_test)

    run_times = np.array(run_times)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)

    assert len(X) != 0
    assert run_times.dtype == np.float, (task_id, run_times.dtype)
    assert len(X) == run_times.shape[0]
    assert Y_train.dtype == np.float, (task_id, Y_train.dtype)
    assert len(X) == Y_train.shape[0]
    assert Y_test.dtype == np.float, (task_id, Y_test.dtype)
    assert len(X) == Y_test.shape[0]

    if len(run_times.shape) == 1:
        raise ValueError()

    # Get all configs with the best value
    incumbent_test = list(np.where(Y_test == Y_test.min())[0])

    # Shuffle incumbent array
    rng = np.random.RandomState(task_id)
    rng.shuffle(incumbent_test)
    for idx in incumbent_test:
        config = Configuration(cs, values=X[idx])
        incumbents_test_rval.append(config)
    # Return all incumbents
    return task_id, incumbents_test_rval


def hash_config(config):
    repr = config.__repr__()
    m = hashlib.md5()
    m.update(repr.encode('utf8'))
    return m.hexdigest()[-7:]


def main(input_dir, method, method_type, output_dir, taskset, nseeds=3):
    assert os.path.isdir(input_dir), input_dir
    cs_file_tmp = os.path.join(input_dir, method_type, method + "*", 'space.json')
    cs_file = glob.glob(cs_file_tmp)
    if len(cs_file) == 0:
        print("Could not find space %s" % cs_file_tmp)
    with open(cs_file[0]) as fh:
        cs = read(fh.read())

    task_id_to_dir = defaultdict(list)
    incumbents_test = list()
    config_to_tasks = defaultdict(list)

    for task_id in taskset:
        configuration_output_dir = os.path.join(
            input_dir,
            method_type,
            '%s_%d_*_0_0' % (method, task_id),
            'auto-sklearn-output',
            'smac3-output',
            'run_*',
            'runhistory.json',
        )
        configuration_output_dirs = glob.glob(configuration_output_dir)
        if len(configuration_output_dirs) != nseeds:
            print("Skip", configuration_output_dir, "has only", len(configuration_output_dirs), "runhistories")
            continue
        task_id_to_dir[task_id] = configuration_output_dirs

    print("Found", len(task_id_to_dir), "complete entries")
    print("Skipped", len(automl_metadata) - len(task_id_to_dir), "entries")

    rval = Parallel(n_jobs=8, verbose=0)(
        delayed(read_configurations_for_task_id)(task_id, task_id_to_dir, cs)
        for task_id in sorted(list(task_id_to_dir))
    )
    for task_id, ivt in rval:
        for inc in ivt:
            if inc not in incumbents_test:
                incumbents_test.append(inc)
                config_to_tasks[inc].append(task_id)
                break
            else:
                config_to_tasks[inc].append(task_id)
    print("Found", len(incumbents_test), "incumbents")

    jason = {hash_config(i): i for i in incumbents_test}
    assert len(jason) == len(incumbents_test)

    drop_keys = set()
    for idx, i in enumerate(jason):
        for jdx, j in enumerate(jason):
            if idx >= jdx:
                continue
            else:
                if jason[i].get_dictionary() == jason[j].get_dictionary():
                    drop_keys.add(j)
    for key in drop_keys:
        raise ValueError("Found double entry:", jason[key])
        del jason[key]

    config_id_to_task = dict()
    for key in jason:
        config_id_to_task[key] = list(config_to_tasks[jason[key]])
        jason[key] = jason[key].get_dictionary()

    print('Found %d incuments!' % len(jason))

    json_file_name = os.path.join(output_dir, 'incumbents.json')
    with open(json_file_name, 'w') as fh:
        json.dump(jason, fh, indent=4)

    json_file_name = os.path.join(output_dir, 'task_to_inc_id.json')
    with open(json_file_name, 'w') as fh:
        json.dump(config_id_to_task, fh, indent=4)

    configspace_file_name = os.path.join(output_dir, 'space.json')
    with open(configspace_file_name, 'w') as fh:
        fh.write(write(cs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--method', choices=list(method_dc.keys()), required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--nseeds', type=int, required=True)
    parser.add_argument('--taskset', choices=list(dataset_dc.keys()), required=True)
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    method = args.method
    output_dir = os.path.join(args.output_dir, method)
    if os.path.exists(output_dir):
        raise ValueError("Nothing left to do!")
    else:
        os.makedirs(output_dir)

    if method.startswith("RF"):
        method_type = "RF"
    elif method.startswith("None"):
        method_type = "None"
    else:
        raise ValueError("Don't know method %s" % method)
    out_dir = "./"
    main(input_dir=input_dir, method=method, taskset=dataset_dc[args.taskset],
         method_type=method_type, output_dir=output_dir, nseeds=args.nseeds)
