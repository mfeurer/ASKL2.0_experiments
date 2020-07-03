import argparse
import copy
import os
import sys

import numpy as np
import sklearn.model_selection

sys.path.append('.')
sys.path.append('..')
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--input-directory', type=str, required=True)
parser.add_argument('--n-seeds', type=int, required=True)
parser.add_argument('--max-runtime', type=int)
parser.add_argument('--portfolio-size', type=int, required=True)
parser.add_argument('--output-directory', type=str, required=True)
parser.add_argument('--max-execution-runtime', type=int, required=True)


args = parser.parse_args()
input_directory = args.input_directory
n_seeds = args.n_seeds
portfolio_size = args.portfolio_size
max_runtime = args.max_runtime
output_directory = args.output_directory
max_execution_runtime = args.max_execution_runtime

input_directory = os.path.expanduser(input_directory)
output_directory = os.path.expanduser(output_directory)
os.makedirs(output_directory, exist_ok=True)
portfolio_execution_output_directory = os.path.join(output_directory, 'portfolio_execution')
os.makedirs(portfolio_execution_output_directory, exist_ok=True)

call_template = (
    'python /home/feurerm/projects/2020_IEEE_Autosklearn_experiments/experiment_scripts/portfolio/build_portfolio.py '
    '--input-directory %s '
    '--portfolio-size %d '
) % (input_directory, portfolio_size)
if max_runtime is not None:
    call_template = call_template + (' --max-runtime %d' % max_runtime)


metadata_task_ids = np.array(utils.automl_metadata, dtype=int)


commands = []
commands_cv = []
execute_portfolio_calls = []
for fidelities in ['NoFidelity', 'SH']:
    for seed in range(n_seeds):

        call = copy.copy(call_template)

        output_file = os.path.join(
            output_directory,
            '%d_%s_%s_%d.json' % (portfolio_size, fidelities, max_runtime, seed)
        )

        call = call + (' --fidelities %s' % fidelities)
        call = call + (' --seed %d' % seed)
        main_call = call + (' --output-file %s' % output_file)
        commands.append(main_call)

        kfold = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=seed)
        for split_ixd, (training_indices, test_indices) in enumerate(kfold.split(metadata_task_ids)):
            training_task_ids = metadata_task_ids[training_indices]
            test_task_ids = metadata_task_ids[test_indices]

            output_file = os.path.join(
                output_directory,
                '%d_%s_%s_%d_%d.json' % (portfolio_size, fidelities, max_runtime, seed, split_ixd)
            )

            tmp_call = call + ' --training-task-ids '
            tmp_call = tmp_call + ' '.join([str(tti) for tti in training_task_ids])
            tmp_call = tmp_call + ' --output-file %s' % output_file
            commands_cv.append(tmp_call)

            for task_id in test_task_ids:
                portfolio_execution_output_file = os.path.join(
                    portfolio_execution_output_directory,
                    '%d_%s_%s_%d_%d.json' % (portfolio_size, fidelities, max_runtime, seed, task_id)
                )

                execute_portfolio_call = (
                    'python /home/feurerm/projects/2020_IEEE_Autosklearn_experiments/experiment_scripts/portfolio/execute_portfolio.py '
                    '--portfolio-file %s '
                    '--input-directory %s '
                    '--output-file %s '
                    '--fidelities %s '
                    '--task-id %d '
                    '--max-runtime %d'
                ) % (output_file, input_directory, portfolio_execution_output_file,
                     fidelities, int(task_id), max_execution_runtime)
                execute_portfolio_calls.append(execute_portfolio_call)


call_file = os.path.join(output_directory, 'build_portfolio_%d_%s.cmd' % (portfolio_size, max_runtime))
with open(call_file, 'wt') as fh:
    fh.write('\n'.join(commands))
cv_call_file = os.path.join(output_directory, 'build_portfolio_cv_%d_%s.cmd' % (portfolio_size, max_runtime))
with open(cv_call_file, 'wt') as fh:
    fh.write('\n'.join(commands_cv))
execute_portfolio_call_file = os.path.join(output_directory, 'execute_portfolio_cv_%d_%s.cmd' % (portfolio_size, max_runtime))
with open(execute_portfolio_call_file, 'wt') as fh:
    fh.write('\n'.join(execute_portfolio_calls))
