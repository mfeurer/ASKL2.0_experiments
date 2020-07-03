import argparse
import json
import os
import pickle
import sys
import time

from ConfigSpace.read_and_write.json import read as cs_read_json
import numpy as np

sys.path.append('.')
import greedy_portfolio
import portfolio_util


parser = argparse.ArgumentParser()
parser.add_argument('--input-directory', type=str, required=True)
parser.add_argument('--output-file', type=str, required=True)
parser.add_argument('--portfolio-size', type=int, required=True)
parser.add_argument('--fidelities', choices=['None', 'SH'], required=True)
parser.add_argument('--training-task-ids', type=int, nargs='*')
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--max-runtime', type=int)

args = parser.parse_args()
input_directory = args.input_directory
output_file = args.output_file
portfolio_size = args.portfolio_size
fidelities = args.fidelities
training_task_ids = args.training_task_ids
seed = args.seed
rng = np.random.RandomState(seed)
max_runtime = args.max_runtime

if training_task_ids is not None:
    task_ids = []
    for task_id in training_task_ids:
        if task_id not in portfolio_util._training_task_ids:
            raise ValueError('training_task_id %d does not exist!' % task_id)
        else:task_ids.append(task_id)
else:
    task_ids = portfolio_util._training_task_ids

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

start_time = time.time()
portfolio_member_ids, cutoffs, budget_to_idx = greedy_portfolio.build(
    matrix=matrix,
    task_ids=task_ids,
    configurations=incumbents,
    portfolio_size=portfolio_size,
    max_runtime=max_runtime,
    rng=rng,
    fidelity_strategy_name=fidelities,
    fidelity_strategy_kwargs=fidelity_strategy_kwargs,
)
end_time = time.time()

portfolio = {
    config_id: incumbents[config_id] for config_id in portfolio_member_ids
}
config_to_task_subset = {
    key: value for key, value in incumbent_to_task.items() if key in portfolio
}
cutoffs = [float(cutoff) for cutoff in cutoffs]

output_json = {
    'portfolio': portfolio,
    'cutoffs': cutoffs,
    'budget_to_idx': budget_to_idx,
    'config_to_task': config_to_task_subset,
    'input_directory': input_directory,
    'fidelities': fidelities,
    'portfolio_size': portfolio_size,
    'seed': seed,
    'max_runtime': max_runtime,
    'start_time': start_time,
    'end_time': end_time,
    'wallclock_time': end_time - start_time,
}

with open(output_file, 'wt') as fh:
    json.dump(output_json, fh, indent=4)
