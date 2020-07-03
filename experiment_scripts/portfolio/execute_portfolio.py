import argparse
import json
import os
import pickle
import sys

from ConfigSpace.read_and_write.json import read as cs_read_json
import numpy as np

import sys
sys.path.append('.')

import portfolio_util
import fidelity_strategies


parser = argparse.ArgumentParser()
parser.add_argument('--portfolio-file', type=str, required=True)
parser.add_argument('--input-directory', type=str, required=True)
parser.add_argument('--output-file', type=str, required=True)
parser.add_argument('--fidelities', choices=['None', 'SH'], required=True)
parser.add_argument('--task-id', type=int)
parser.add_argument('--max-runtime', type=int, required=True)

args = parser.parse_args()
portfolio_file = args.portfolio_file
input_directory = args.input_directory
output_file = args.output_file
fidelities = args.fidelities
task_id = args.task_id
max_runtime = args.max_runtime

with open(portfolio_file) as fh:
    portfolio = json.load(fh)

with open(os.path.join(input_directory, 'incumbents.json')) as fh:
    incumbents = json.load(fh)

with open(os.path.join(input_directory, 'space.json')) as fh:
    space = cs_read_json(fh.read())

with open(os.path.join(input_directory, 'task_to_inc_id.json')) as fh:
    incumbent_to_task = json.load(fh)

with open(os.path.join(input_directory, 'matrix.pkl'), 'rb') as fh:
    matrix = pickle.load(fh)

if fidelities == 'None':
    fidelity_strategy_kwargs = {}
elif fidelities == 'SH':
    fidelity_strategy_kwargs = {'eta': 4, 'min_budget': 6.25, 'max_budget': 100}
else:
    raise ValueError(fidelities)

y_valid, y_test, runtimes, config_id_to_idx, task_id_to_idx = portfolio_util.reformat_data(
    matrix, portfolio_util._training_task_ids, incumbents)
normalized_matrix = portfolio_util.normalize_matrix(y_test)

fidelity_strategy = fidelity_strategies.build_fidelity_strategy(
    fidelity_strategy_name=fidelities,
    kwargs=fidelity_strategy_kwargs,
)

task_idx = task_id_to_idx[task_id]
play_kwargs = dict(
    y_valid=y_valid[task_idx],
    y_test=y_test[task_idx],
    runtimes=runtimes[task_idx],
    configurations=list(portfolio['portfolio']),
    config_id_to_idx=config_id_to_idx,
    cutoffs=portfolio['cutoffs'],
    max_runtime=max_runtime,
)

budget_to_idx = [{float(key): value for key, value in entry.items()}
                 for entry in portfolio['budget_to_idx']]
play_kwargs['config_to_budget_to_idx'] = {
    config_id_to_idx[config_id]: budget_to_idx[i]
    for i, config_id in enumerate(portfolio['portfolio'])
}

min_val_score, min_test_score, trajectory = fidelity_strategy.play(**play_kwargs)

trajectory = trajectory.to_dict()
trajectory = [[key, value] for key, value in trajectory.items()]

result = {}
result[0] = {
    'task_id': task_id,
    'time_limit': max_runtime,
    'loss': min_test_score,
    'trajectory': trajectory,
}

with open(output_file, 'wt') as fh:
    json.dump(result, fh, indent=4)
