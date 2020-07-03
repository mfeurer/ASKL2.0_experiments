import argparse
import collections
import json
import glob
import os
import pickle
import sys

import numpy as np
import openml
import pandas as pd

this_dir = os.path.dirname(__file__)
main_dir = os.path.abspath(os.path.join(this_dir, '..'))
sys.path.append(main_dir)
import utils
import autoauto_static
import autoauto_dynamic


parser = argparse.ArgumentParser()
parser.add_argument('--metadata-type',
                    choices=['portfolio', 'full_runs', 'full_runs_ensemble'],
                    default='portfolio')
parser.add_argument('--input-directories', type=str, required=True, nargs='+')
parser.add_argument('--matrices-directory', type=str, required=True)
parser.add_argument('--metafeatures-directory', type=str, required=True)
parser.add_argument('--output-file', type=str, required=True)
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--portfolios-with-max-runtime', action='store_true')
parser.add_argument('--type', choices=['static', 'dynamic'])
parser.add_argument('--only', nargs='*')

args = parser.parse_args()
metadata_type = args.metadata_type
input_directories = args.input_directories
matrices_directory = args.matrices_directory
metafeatures_directory = args.metafeatures_directory
output_file = args.output_file
seed = args.seed
rng = np.random.RandomState(seed)
portfolios_with_max_runtime = args.portfolios_with_max_runtime
autoauto_type = args.type
only = args.only

# Zeroth, get all meta-features
# meta_features = dict()
# for task_id in utils.openml_automl_benchmark:
#     meta_features[task_id] = utils.get_meta_features(
#             task_id, '/home/eggenspk/PoSHAutosklearn/2020_IEEE_Autosklearn_experiments/'
#                      'experiment_scripts/10MIN/AutoAuto_build_full_data_more_metafeatures/metafeatures/')
# meta_features = pd.DataFrame(meta_features)
# meta_features = openml.tasks.list_tasks(task_type_id=1, output_format='dataframe')
# meta_features = meta_features.loc[utils.automl_metadata]
# meta_features = meta_features[
#     ['NumberOfClasses',
#      'NumberOfFeatures',
#      'NumberOfInstances',
#      'MinorityClassSize',
#      'MajorityClassSize',
#      'MaxNominalAttDistinctValues',
#      'NumberOfInstancesWithMissingValues',
#      'NumberOfMissingValues',
#      'NumberOfNumericFeatures',
#      'NumberOfSymbolicFeatures',
#      ]
# ]
# meta_features['class_ration'] = meta_features['MinorityClassSize'] / meta_features['MajorityClassSize']
# meta_features.fillna(0, inplace=True)
# meta_features = meta_features.astype(float)
meta_features = dict()
for task_id in utils.automl_metadata:
    meta_features[task_id] = utils.get_meta_features(task_id, metafeatures_directory)
meta_features = pd.DataFrame(meta_features).transpose()
meta_features = meta_features[
    ['NumberOfClasses',
     'NumberOfFeatures',
     'NumberOfInstances',
#     'MinorityClassSize',
#     'MajorityClassSize',
#     'MaxNominalAttDistinctValues',
#     'NumberOfInstancesWithMissingValues',
#     'NumberOfMissingValues',
#     'NumberOfNumericFeatures',
#     'NumberOfSymbolicFeatures',
     ]
]
#meta_features['class_ration'] = meta_features['MinorityClassSize'] / meta_features['MajorityClassSize']
meta_features.fillna(0, inplace=True)
meta_features = meta_features.astype(float)
print(meta_features)

methods_to_choose_from = dict()
if metadata_type == 'portfolio':
    max_runtime_ = None
    # First, get all methods which can be found in the input directories

    for input_directory in input_directories:
        glop_path = os.path.join(input_directory, '*_*_*_%d_*.json' % seed)
        files = glob.glob(glop_path)
        for filepath in files:
            full_directory = os.path.split(filepath)[0]
            method_key = os.path.split(filepath)[0].split('/')[-1]
            filename = os.path.split(filepath)[1]
            filename = filename.replace('.json', '')
            filename = filename.split('_')
            max_runtime = filename[2]
            if max_runtime == 'None' and portfolios_with_max_runtime:
                continue
            elif max_runtime != 'None' and not portfolios_with_max_runtime:
                continue
            max_runtime_ = max_runtime
            path_template = os.path.join(
                full_directory,
                'portfolio_execution',
                ('_'.join(filename[:4])) + '_%d.json'
            )
            portfolio_file = os.path.join(full_directory, '_'.join(filename[:4]) + '.json')
            portfolio_file = portfolio_file.replace('portfolio_execution/', '')
            methods_to_choose_from[method_key] = {'path_template': path_template,
                                                  'portfolio_file': portfolio_file, }
    print('Max runtime of the portfolio for learning', max_runtime_)
else:
    if len(input_directories) != 1:
        raise ValueError('Requires exactly one input directory, contains %d' % len(input_directories))
    input_directory = input_directories[0]
    if metadata_type == 'full_runs':
        glop_path = os.path.join(input_directory, '*_%d_0_0' % seed, 'result.json')
    else:
        glop_path = os.path.join(input_directory, '*_%d_0_0' % seed, 'ensemble_results_0.000000thresh_50size_1.000000best')
    files = glob.glob(glop_path)
    if len(files) == 0:
        print(glop_path)
        raise ValueError('Could not find a result file at %s' % glop_path)
    for filepath in files:
        full_directory = os.path.split(filepath)[0]
        method_key = os.path.split(filepath)[0].split('/')[-1]
        method_key = method_key.split('_')
        method = '_'.join(method_key[:-4])
        path_template = os.path.join(
            input_directory,
            method + '_%d_' + str(method_key[-3]) + '_0_0',
            'result.json' if metadata_type == 'full_runs' else 'ensemble_results_0.000000thresh_50size_1.000000best'
        )
        methods_to_choose_from[method] = {'path_template': path_template}

keys = list(methods_to_choose_from.keys())

if only:
    print('Dropping keys, was', keys)
    keys = [key for key in keys if key in only]

keys = sorted(keys)
keys = list(rng.permutation(keys))
methods_to_choose_from = {key: methods_to_choose_from[key] for key in keys}
print(methods_to_choose_from.keys())


# Second, load the cross-validated performance matrices
performance_matrix = collections.defaultdict(dict)
for method, additional_info in methods_to_choose_from.items():
    for task_id in utils.automl_metadata:
        if task_id in (2121, 189829):
            # These two tasks don't work at the moment
            performance_matrix[method][task_id] = 1.0
            continue
        path = additional_info['path_template'] % task_id
        with open(path) as fh:
            results = json.load(fh)
            if '0' in results:
                loss = results['0']['loss']
            else:
                loss = results['50']['loss']
            loss = loss if np.isfinite(loss) else 1.0
            performance_matrix[method][task_id] = loss
performance_matrix = pd.DataFrame(performance_matrix)

# Third, load the performance matrices from which the portfolios were built
matrices = {}
incumbents = {}
for method in methods_to_choose_from:
    model_selection_strategy = utils.method_dc[method]['evaluation']
    if model_selection_strategy == 'CV':
        model_selection_strategy = '%dCV' % utils.method_dc[method]['cv']
    matrix_path = os.path.join(
        matrices_directory,
        'RF_None_%s_iterative_es_if' % model_selection_strategy,
        'matrix.pkl'
    )
    with open(matrix_path, 'rb') as fh:
        matrices[method] = pickle.load(fh)
    with open(os.path.join(
            matrices_directory,
            'RF_None_%s_iterative_es_if' % model_selection_strategy,
            'incumbents.json')
    ) as fh:
        incumbents[method] = list(json.load(fh))

if len(matrices) == 0:
    raise ValueError('Could not find a matrix to load')


if autoauto_type == 'static':
    selector = autoauto_static.build(
        performance_matrix=performance_matrix,
        matrices=matrices,
        metafeatures=meta_features,
        random_state=rng,
        task_ids=utils.automl_metadata,
        configurations=incumbents,
        seed=seed,
    )
elif autoauto_type == 'dynamic':
    selector = autoauto_dynamic.build(
        performance_matrix=performance_matrix,
        matrices=matrices,
        metafeatures=meta_features,
        random_state=rng,
        task_ids=utils.automl_metadata,
        configurations=incumbents,
        seed=seed,
    )
else:
    raise ValueError(autoauto_type)

to_pickle = {
    'selector': selector,
    # Store this as an additional list to make sure sorting is preserved!
    'methods_to_choose_from': [method for method in methods_to_choose_from],
    'methods_information': methods_to_choose_from,
}

output_dir = os.path.dirname(output_file)
os.makedirs(output_dir, exist_ok=True)

with open(output_file, 'wb') as fh:
    pickle.dump(to_pickle, fh)
