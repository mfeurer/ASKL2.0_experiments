import lockfile
import glob
import json
import os
from typing import Dict

import numpy as np
import pandas as pd
import pickle

import openml


RF = ["RF_None_holdout_iterative_es_if",
      "RF_None_3CV_iterative_es_if",
      "RF_None_5CV_iterative_es_if",
      "RF_None_10CV_iterative_es_if"]
RFSH = ["RF_SH-eta4-i_holdout_iterative_es_if",
        "RF_SH-eta4-i_3CV_iterative_es_if",
        "RF_SH-eta4-i_5CV_iterative_es_if",
        "RF_SH-eta4-i_10CV_iterative_es_if"]
IMP0 = [
    # ifnif
    "RF_None_holdout_iterative_es_if", "RF_None_holdout_iterative_es_nif",
    "RF_None_holdout_full_es_if", "RF_None_holdout_full_es_nif",
    # searchspace
    "RF_None_holdout_iterative-preproc_es_if"
]

ASKL_FULL = ["RF_None_holdout_full_es_nif",
            ]
ASKL_FULL_RANDOM = ["None_None_holdout_full_es_nif",
                   ]
ASKL_ITER = ["RF_None_holdout_iterative_es_nif",
            ]
ASKL_ITER_RANDOM = ["None_None_holdout_iterative_es_nif",
                   ]


openml_cc18_ids = [
    167149, 167150, 167151, 167152, 167153, 167154, 167155, 167156, 167157, 167158, 167159, 167160, 167161, 167162,
    167163, 167165, 167166, 167167, 167168, 167169, 167170, 167171, 167164, 167173, 167172, 167174, 167175, 167176,
    167177, 167178, 167179, 167180, 167181, 167182, 126025, 167195, 167194, 167190, 167191, 167192, 167193, 167187,
    167188, 126026, 167189, 167185, 167186, 167183, 167184, 167196, 167198, 126029, 167197, 126030, 167199, 126031,
    167201, 167205, 189904, 167106, 167105, 189905, 189906, 189907, 189908, 189909, 167083, 167203, 167204, 189910,
    167202, 167097
]

# 33% Holdout tasks from automl benchmark set: https://www.openml.org/s/218
# Not using did 2 and 5 from this study as data on openml is wrong for the automl benchmark
openml_automl_benchmark = [
    189871, 189872, 189873, 168794, 168792, 168793, 75105, 189906, 189909, 189908, 167185, 189874, 189861, 189866,
    168797, 168796, 189860, 189862, 168798, 189865, 126026, 167104, 167083, 189905, 75127, 167200, 167184, 167201,
    168795, 126025, 75097, 167190, 126029, 167149, 167152, 167168, 167181, 75193, 167161
]

automl_metadata = [
    232, 236, 241, 245, 253, 254, 256, 258, 260, 262, 267, 271, 273, 275, 279, 288, 336, 340, 2119, 2120, 2121, 2122,
2123, 2125, 2356, 3044, 3047, 3048, 3049, 3053, 3054, 3055, 75089, 75092, 75093, 75098, 75100, 75108, 75109, 75112,
75114, 75115, 75116, 75118, 75120, 75121, 75125, 75126, 75129, 75131, 75133, 75134, 75136, 75139, 75141, 75142,
75143, 75146, 75147, 75148, 75149, 75153, 75154, 75156, 75157, 75159, 75161, 75163, 75166, 75169, 75171, 75173,
75174, 75176, 75178, 75179, 75180, 75184, 75185, 75187, 75192, 75195, 75196, 75199, 75210, 75212, 75213, 75215,
75217, 75219, 75221, 75223, 75225, 75232, 75233, 75234, 75235, 75236, 75237, 75239, 75250, 126021, 126024, 126028,
126030, 126031, 146574, 146575, 146576, 146577, 146578, 146583, 146586, 146592, 146593, 146594, 146596, 146597,
146600, 146601, 146602, 146603, 146679, 166859, 166866, 166872, 166875, 166882, 166897, 166905, 166906, 166913,
166915, 166931, 166932, 166944, 166950, 166951, 166953, 166956, 166957, 166958, 166959, 166970, 166996, 167085,
167086, 167087, 167088, 167089, 167090, 167094, 167096, 167097, 167099, 167100, 167101, 167103, 167105, 167106,
167202, 167203, 167204, 167205, 168785, 168791, 189779, 189786, 189828, 189829, 189836, 189840, 189841, 189843,
189844, 189845, 189846, 189857, 189858, 189859, 189863, 189864, 189869, 189870, 189875, 189878, 189880, 189881,
189882, 189883, 189884, 189887, 189890, 189893, 189894, 189899, 189900, 189902, 190154, 190155, 190156, 190157,
190158, 190159, 211720, 211721, 211722, 211723, 211724
]

#Adult (binary, as in caruana)
#Australian (binary, as it is the smallest dataset)
#Covertype (multiclass, as in caruana, is also the largest dataset)
#guillermo (binary, an actual AutoML2 challenge dataset)
#jungle chess complete (multiclass)
#kc1 (binary, appears to allow for a lot of overfitting)
#KDDCup09_appetency (binary, appears to be a nice dataset)
#MiniBooNE (binary, another largish dataset, but we know the origin compared to the challenge datasets)
ensemble_datasets = [
    126025, 167104, 75193, 168796, 189909, 167181, 75105, 168798
]

ensemble_mini_datasets = [
    126025, 168796,
]

dataset_dc = {
    "openml_cc18_ids": openml_cc18_ids,
    "openml_automl_benchmark": openml_automl_benchmark,
    "automl_metadata": automl_metadata,
    "ensemble_datasets": ensemble_datasets,
    "ensemble_mini_datasets": ensemble_mini_datasets,
}

def load_task(task_id):
    task = openml.tasks.get_task(task_id)
    X, y = task.get_X_and_y()
    train_indices, test_indices = task.get_train_test_split_indices()
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    dataset = openml.datasets.get_dataset(task.dataset_id)
    _, _, cat, _ = dataset.get_data(target=task.target_name)
    del _
    del dataset
    cat = ['categorical' if c else 'numerical' for c in cat]

    return X_train, y_train, X_test, y_test, cat

def get_meta_features(task_id, tmp_dir):
    meta_features_file = os.path.join(tmp_dir, '%s.json' % task_id)
    if os.path.exists(meta_features_file):
        with lockfile.LockFile(meta_features_file):
            with open(meta_features_file) as fh:
                meta_features = json.load(fh)
    else:
        print('Computing metafeatures for task', task_id)
        try:
            os.makedirs(tmp_dir)
        except:
            pass
        #meta_features = openml.tasks.get_task(task_id).get_dataset().qualities
        X, y, _, _, _ = load_task(task_id)
        meta_features = compute_meta_features(X, y)
        with lockfile.LockFile(meta_features_file):
            with open(meta_features_file, 'wt') as fh:
                json.dump(meta_features, fh, indent=4)
    return meta_features

def compute_meta_features(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    meta_features = {}
    meta_features['NumberOfInstances'] = X.shape[0]
    meta_features['NumberOfFeatures'] = X.shape[1]
    meta_features['NumberOfClasses'] = len(np.unique(y))
    return meta_features

def get_normalization_constants(results_dir, load=False):
    fname = os.path.join(os.path.dirname(__file__), "./norm.pkl")
    if load is True:
        if os.path.isfile(fname):
            with open(fname, "rb") as fh:
                print("Loading")
                return pickle.load(fh)
        else:
            print("%s does not exist" % fname)

    keys_to_load = [
        '10MIN/ASKL_automldata/RF/RF_None_holdout_iterative_es_if',
        '10MIN/ASKL_automldata/RF/RF_None_3CV_iterative_es_if',
        '10MIN/ASKL_automldata/RF/RF_None_5CV_iterative_es_if',
        '10MIN/ASKL_automldata/RF/RF_None_10CV_iterative_es_if',
        '20MIN/ASKL_automldata/RF/RF_None_holdout_iterative_es_if',
        '20MIN/ASKL_automldata/RF/RF_None_3CV_iterative_es_if',
        '20MIN/ASKL_automldata/RF/RF_None_5CV_iterative_es_if',
        '20MIN/ASKL_automldata/RF/RF_None_10CV_iterative_es_if',
        '60MIN/ASKL_automldata/RF/RF_None_holdout_iterative_es_if',
        '60MIN/ASKL_automldata/RF/RF_None_3CV_iterative_es_if',
        '60MIN/ASKL_automldata/RF/RF_None_5CV_iterative_es_if',
        '60MIN/ASKL_automldata/RF/RF_None_10CV_iterative_es_if',
        '10H/ASKL_automldata/RF/RF_None_holdout_iterative_es_if',
        '10H/ASKL_automldata/RF/RF_None_3CV_iterative_es_if',
        '10H/ASKL_automldata/RF/RF_None_5CV_iterative_es_if',
        '10H/ASKL_automldata/RF/RF_None_10CV_iterative_es_if',
    ]

    task_ids = openml_automl_benchmark

    normalization_data = {}
    miss = 0
    for tid in task_ids:
        normalization_data[tid] = {}
        for mode in keys_to_load:
            normalization_data[tid][mode] = []
            for seed in range(10):
                fl_tmpl = results_dir + "/" + mode + "_%d_%d_0_0/auto-sklearn-output/*/*/runhistory.json" % (tid, seed)
                fl = glob.glob(fl_tmpl)
                if len(fl) == 0:
                    continue
                fl = fl[0]
                with open(fl, "r") as fh:
                    line = json.load(fh)
                    line = line["data"]
                    test_losses = []
                    for i in range(len(line)):
                        try:
                            # was this a crash?
                            test_loss = line[i][1][3]["test_loss"]
                        except:
                            test_loss = 1
                        test_losses.append(test_loss)
                    normalization_data[tid][mode].append(test_losses)

    print("Missing %d entries" % miss)

    # First get min and diff values across all datasets
    min_diff_dc = {}
    for tid in task_ids:
        min_for_task = 1.0
        max_for_task = 0.0
        for mode in normalization_data[tid]:
            tmp = pd.DataFrame(normalization_data[tid][mode]).sort_index(axis=1).ffill(axis=1)
            mini = tmp.min().min()
            min_for_task = min(min_for_task, mini)
            maxi = tmp.max().max()
            max_for_task = max(max_for_task, maxi)
        diff = max_for_task - min_for_task
        if diff == 0.0:
            diff = 1.0
        min_diff_dc[tid] = (min_for_task, diff)

    if not os.path.exists(fname):
        with open(fname, "wb") as fh:
            print("Dumped")
            pickle.dump(min_diff_dc, fh)

    return min_diff_dc

def add_classifier_wo_early_stopping():
    import autosklearn.pipeline.components.classification
    import iterative_models_wo_early_stopping
    autosklearn.pipeline.components.classification.add_classifier(
        iterative_models_wo_early_stopping.GradientBoostingClassifierWOEarlyStopping
    )
    autosklearn.pipeline.components.classification.add_classifier(
        iterative_models_wo_early_stopping.SGDWOEarlyStopping
    )
    autosklearn.pipeline.components.classification.add_classifier(
        iterative_models_wo_early_stopping.PassiveAggressiveWOEarlyStopping
    )

method_dc = {
    'RF_None_holdout_iterative_es_if': {
        "model": "RF",
        "fidelity": None,
        "evaluation": "holdout",
        "searchspace": "iterative",
        "iterative-fit": True,
        "early-stopping": True,
    },
    'RF_None_3CV_iterative_es_if': {
        "model": "RF",
        "fidelity": None,
        "evaluation": "CV",
        "cv": 3,
        "searchspace": "iterative",
        "iterative-fit": True,
        "early-stopping": True,
    },
    'RF_None_5CV_iterative_es_if': {
        "model": "RF",
        "fidelity": None,
        "evaluation": "CV",
        "cv": 5,
        "searchspace": "iterative",
        "iterative-fit": True,
        "early-stopping": True,
    },
    'RF_None_10CV_iterative_es_if': {
        "model": "RF",
        "fidelity": None,
        "evaluation": "CV",
        "cv": 10,
        "searchspace": "iterative",
        "iterative-fit": True,
        "early-stopping": True,
    },
    'RF_SH-eta4-i_3CV_iterative_es_if': {
        "model": "RF",
        "fidelity": "SH",
        "eta": 4,
        "budget-type": "iterations",
        "evaluation": "CV",
        "cv": 3,
        "searchspace": "iterative",
        "iterative-fit": True,
        "early-stopping": True,
    },
    'RF_SH-eta4-i_holdout_iterative_es_if': {
        "model": "RF",
        "fidelity": "SH",
        "eta": 4,
        "budget-type": "iterations",
        "evaluation": "holdout",
        "searchspace": "iterative",
        "iterative-fit": True,
        "early-stopping": True,
    },
    'RF_SH-eta4-i_5CV_iterative_es_if': {
        "model": "RF",
        "fidelity": "SH",
        "eta": 4,
        "budget-type": "iterations",
        "evaluation": "CV",
        "cv": 5,
        "searchspace": "iterative",
        "iterative-fit": True,
        "early-stopping": True,
    },
    'RF_SH-eta4-i_10CV_iterative_es_if': {
        "model": "RF",
        "fidelity": "SH",
        "eta": 4,
        "budget-type": "iterations",
        "evaluation": "CV",
        "cv": 10,
        "searchspace": "iterative",
        "iterative-fit": True,
        "early-stopping": True,
    },
    'RF_None_holdout_full_es_nif': {
        "model": "RF",
        "fidelity": None,
        "evaluation": "holdout",
        "searchspace": "full",
        "iterative-fit": False,
        "early-stopping": True,
    },
    'None_None_holdout_full_es_nif': {
        "model": None,
        "fidelity": None,
        "evaluation": "holdout",
        "searchspace": "full",
        "iterative-fit": False,
        "early-stopping": True,
    },
    'RF_None_holdout_iterative_es_nif': {
        "model": "RF",
        "fidelity": None,
        "evaluation": "holdout",
        "searchspace": "iterative",
        "iterative-fit": False,
        "early-stopping": True,
    },
    'None_None_holdout_iterative_es_nif': {
        "model": None,
        "fidelity": None,
        "evaluation": "holdout",
        "searchspace": "iterative",
        "iterative-fit": False,
        "early-stopping": True,
    },
    'RF_None_holdout_iterative-preproc_es_if': {
            "model": "RF",
            "fidelity": None,
            "evaluation": "holdout",
            "searchspace": "iterative-preproc",
            "iterative-fit": True,
            "early-stopping": True,
    },
    'RF_None_holdout_full_es_if': {
        "model": "RF",
        "fidelity": None,
        "evaluation": "holdout",
        "searchspace": "full",
        "iterative-fit": True,
        "early-stopping": True,
    },
}
