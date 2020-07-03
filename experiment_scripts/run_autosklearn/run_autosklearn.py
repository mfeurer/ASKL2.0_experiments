import argparse
import glob
import gzip
import io
import json
import os
import pickle
import shutil
import subprocess
import tempfile

import autosklearn
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.metrics import balanced_accuracy
from ConfigSpace.read_and_write.json import write
from ConfigSpace import Configuration
import numpy as np
from smac.facade.roar_facade import ROAR
from smac.scenario.scenario import Scenario

import sys
this_dir = os.path.dirname(__file__)
main_dir = os.path.abspath(os.path.join(this_dir, '..'))
sys.path.append(main_dir)
from utils import add_classifier_wo_early_stopping, load_task
import ensembles.run_ensemble_builder

strio = io.StringIO()
completed_process = subprocess.run(
    'git log -1', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
)
strio.write('###\n')
strio.write('Experiment scripts repository - latest commit:\n')
strio.write(str(completed_process.stdout, encoding='utf8'))
strio.write(str(completed_process.stderr, encoding='utf8'))
strio.write('###\n')
strio.write('Auto-sklearn repository - latest commit:\n')
autosklearn_directory = os.path.dirname(autosklearn.__file__)
completed_process = subprocess.run(
    'cd %s; git log -1' % autosklearn_directory,
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True,
)
strio.write(str(completed_process.stdout, encoding='utf8'))
strio.write(str(completed_process.stderr, encoding='utf8'))
strio.write('###\n')
strio.write('Conda list:\n')
completed_process = subprocess.run('conda list', shell=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,)
strio.write(str(completed_process.stdout, encoding='utf8'))
strio.write(str(completed_process.stderr, encoding='utf8'))
strio.seek(0)


def get_smac_object_callback(portfolio):
    def get_smac_object(
        scenario_dict,
        seed,
        ta,
        ta_kwargs,
        backend,
        metalearning_configurations,
    ):
        from smac.facade.smac_ac_facade import SMAC4AC
        from smac.runhistory.runhistory2epm import RunHistory2EPM4LogCost
        from smac.scenario.scenario import Scenario

        scenario_dict['input_psmac_dirs'] = backend.get_smac_output_glob(
            smac_run_id=seed if not scenario_dict['shared-model'] else '*',
        )
        scenario = Scenario(scenario_dict)
        if len(metalearning_configurations) > 0:
            default_config = scenario.cs.get_default_configuration()
            initial_configurations = [default_config] + metalearning_configurations
        elif portfolio:
            initial_configurations = [
                Configuration(configuration_space=scenario.cs, values=member)
                for member in portfolio]
        else:
            initial_configurations = None
        rh2EPM = RunHistory2EPM4LogCost
        return SMAC4AC(
            scenario=scenario,
            rng=seed,
            runhistory2epm=rh2EPM,
            tae_runner=ta,
            tae_runner_kwargs=ta_kwargs,
            initial_configurations=initial_configurations,
            run_id=seed,
        )
    return get_smac_object


def get_sh_or_hb_object_callback(budget_type, bandit_strategy, eta, initial_budget, portfolio):
    def get_smac_object(
        scenario_dict,
        seed,
        ta,
        ta_kwargs,
        backend,
        metalearning_configurations,
    ):
        from smac.facade.smac_ac_facade import SMAC4AC
        from smac.intensification.successive_halving import SuccessiveHalving
        from smac.intensification.hyperband import Hyperband
        from smac.runhistory.runhistory2epm import RunHistory2EPM4LogCost
        from smac.scenario.scenario import Scenario

        scenario_dict['input_psmac_dirs'] = backend.get_smac_output_glob(
            smac_run_id=seed if not scenario_dict['shared-model'] else '*',
        )
        scenario = Scenario(scenario_dict)
        if len(metalearning_configurations) > 0:
            default_config = scenario.cs.get_default_configuration()
            initial_configurations = [default_config] + metalearning_configurations
        elif portfolio:
            initial_configurations = [
                Configuration(configuration_space=scenario.cs, values=member)
                for member in portfolio]
        else:
            initial_configurations = None
        rh2EPM = RunHistory2EPM4LogCost

        ta_kwargs['budget_type'] = budget_type

        if bandit_strategy == 'sh':
            bandit = SuccessiveHalving
        elif bandit_strategy == 'hb':
            bandit = Hyperband
        else:
            raise ValueError(bandit_strategy)

        smac4ac = SMAC4AC(
            scenario=scenario,
            rng=seed,
            runhistory2epm=rh2EPM,
            tae_runner=ta,
            tae_runner_kwargs=ta_kwargs,
            initial_configurations=initial_configurations,
            run_id=seed,
            intensifier=bandit,
            intensifier_kwargs={
                'initial_budget': initial_budget,
                'max_budget': 100,
                'eta': eta,
                'min_chall': 1},
            )
        smac4ac.solver.epm_chooser.min_samples_model = int(len(scenario.cs.get_hyperparameters()) / 2)
        return smac4ac
    return get_smac_object


def get_old_sh_object_callback(budget_type, bandit_strategy, eta, initial_budget):
    def get_smac_object(
        scenario_dict,
        seed,
        ta,
        ta_kwargs,
        backend,
        metalearning_configurations,
    ):
        from smac.facade.smac_ac_facade import SMAC4AC
        from successive_halving import OldSuccessiveHalving
        from smac.runhistory.runhistory2epm import RunHistory2EPM4LogCost
        from smac.scenario.scenario import Scenario
        from smac.tae.execute_ta_run import StatusType

        scenario_dict['input_psmac_dirs'] = backend.get_smac_output_glob(
            smac_run_id=seed if not scenario_dict['shared-model'] else '*',
        )
        scenario = Scenario(scenario_dict)
        if len(metalearning_configurations) > 0:
            default_config = scenario.cs.get_default_configuration()
            initial_configurations = [default_config] + metalearning_configurations
        else:
            initial_configurations = None
        rh2EPM = RunHistory2EPM4LogCost
        r2EPM_kwargs = {
            'consider_for_higher_budgets_state': []
        }

        ta_kwargs['budget_type'] = budget_type

        if bandit_strategy == 'old_sh':
            bandit = OldSuccessiveHalving
        else:
            raise ValueError(bandit_strategy)

        smac4ac = SMAC4AC(
            scenario=scenario,
            rng=seed,
            runhistory2epm=rh2EPM,
            runhistory2epm_kwargs=r2EPM_kwargs,
            tae_runner=ta,
            tae_runner_kwargs=ta_kwargs,
            initial_configurations=initial_configurations,
            run_id=seed,
            intensifier=bandit,
            intensifier_kwargs={
                'initial_budget': initial_budget,
                'max_budget': 100,
                'eta': eta,
                'min_chall': 1,
            },
        )
        smac4ac.solver.epm_chooser.min_samples_model = int(len(scenario.cs.get_hyperparameters()) / 2)
        return smac4ac
    return get_smac_object


def get_random_search_for_sh_or_hb_object_callback(budget_type, bandit_strategy, eta, initial_budget):
    def get_random_search_for_sh_callback(
            scenario_dict,
            seed,
            ta,
            ta_kwargs,
            backend,
            metalearning_configurations,
    ):
        from smac.intensification.successive_halving import SuccessiveHalving
        from smac.intensification.hyperband import Hyperband
        from smac.scenario.scenario import Scenario
        """Random search."""
        scenario_dict['input_psmac_dirs'] = backend.get_smac_output_glob()
        scenario_dict['minR'] = len(scenario_dict['instances'])
        scenario_dict['initial_incumbent'] = 'RANDOM'
        scenario = Scenario(scenario_dict)

        ta_kwargs['budget_type'] = budget_type

        if bandit_strategy == 'sh':
            bandit = SuccessiveHalving
        elif bandit_strategy == 'hb':
            bandit = Hyperband
        else:
            raise ValueError(bandit_strategy)

        return ROAR(
            scenario=scenario,
            rng=seed,
            tae_runner=ta,
            tae_runner_kwargs=ta_kwargs,
            run_id=seed,
            intensifier=bandit,
            intensifier_kwargs={
                'initial_budget': initial_budget,
                'max_budget': 100,
                'eta': eta,
                'min_chall': 1},
        )
    return get_random_search_for_sh_callback


def get_random_search_object_callback(
        scenario_dict,
        seed,
        ta,
        ta_kwargs,
        backend,
        metalearning_configurations,
):
    """Random search."""
    scenario_dict['input_psmac_dirs'] = backend.get_smac_output_glob()
    scenario_dict['minR'] = len(scenario_dict['instances'])
    scenario_dict['initial_incumbent'] = 'RANDOM'
    scenario = Scenario(scenario_dict)
    return ROAR(
        scenario=scenario,
        rng=seed,
        tae_runner=ta,
        tae_runner_kwargs=ta_kwargs,
        run_id=seed,
    )

parser = argparse.ArgumentParser()
parser.add_argument('--working-directory', type=str, required=True)
parser.add_argument('--overwrite-output-dir-name', type=str)
parser.add_argument('--time-limit', type=int, required=True)
parser.add_argument('--per-run-time-limit', type=int)
parser.add_argument('--task-id', type=int, required=True)
parser.add_argument('-s', '--seed', type=int, required=True)
parser.add_argument('--model', choices=("None", "RF"), required=True)
parser.add_argument('--fidelity', choices=("None", "SH"), required=True)
parser.add_argument('--eta', choices=(3, 4), type=int, required=False)  # depends
parser.add_argument("--budget-type", choices=["subsample", "iterations"], required=False)  # depends
parser.add_argument('--evaluation', choices=("holdout", "CV"), required=True)
parser.add_argument('--cv', choices=(3, 5, 10), type=int, required=False)  # depends
parser.add_argument('--searchspace', choices=("full", "iterative", "iterative-preproc"), required=True)
parser.add_argument("--iterative-fit", choices=("True", "False"), required=True)
parser.add_argument("--early-stopping", choices=("True", "False"), required=True)
parser.add_argument("--posthoc-ensembles", action="store_true")

#parser.add_argument('--searchspace',
#                    choices=["full", "gb", "only-iterative", "not-iterative",
#                             "only-iterative-nopreproc", "only-iterative-cheappreproc",
#                             ],
#                    required=True)
#parser.add_argument('--mode', choices=['smac', 'sh', 'random', 'hb', 'random_hb', 'random_sh',
#                                       'old_sh', 'cv'], required=True)
parser.add_argument('--memory-limit', type=int, required=True)
parser.add_argument('--initial-configurations-via-metalearning', type=int, default=0)
parser.add_argument('--metadata-directory', type=str)
parser.add_argument('--portfolio-file', type=str)
parser.add_argument('--keep-predictions', default=False, action="store_true")
args = parser.parse_args()

# Setting
working_directory = args.working_directory
overwrite_output_dir_name = args.overwrite_output_dir_name
time_limit = args.time_limit
per_run_time_limit = args.per_run_time_limit
task_id = args.task_id
seed = args.seed
memory_limit = args.memory_limit
initial_configurations_via_metalearning = args.initial_configurations_via_metalearning
metadata_directory = args.metadata_directory
portfolio_file = args.portfolio_file
keep_predictions = args.keep_predictions

portfolio = None
if portfolio_file:
    if not os.path.exists(portfolio_file):
        raise ValueError('portfolio file %s does not exist' % portfolio_file)
    elif initial_configurations_via_metalearning:
        raise ValueError('Cannot use both a portfolio and auto-sklearn meta-learning')
    else:
        with open(portfolio_file) as fh:
            portfolio_json = json.load(fh)
            portfolio_cutoff = portfolio_json['cutoffs'][0]
            if not np.isinf(portfolio_cutoff) and per_run_time_limit:
                raise ValueError('Cannot use both a portfolio cutoff and per run time limit.')
            elif np.isinf(portfolio_cutoff):
                per_run_time_limit = per_run_time_limit
            else:
                per_run_time_limit = int(portfolio_cutoff)
            portfolio = list(portfolio_json['portfolio'].values())
elif not per_run_time_limit:
    raise ValueError('Must give a per run time limit')

# Method description
model = args.model if args.model != "None" else None
fidelity = args.fidelity if args.fidelity != "None" else None
eta, budget_type, cv = (None, None, None)
if fidelity == "SH":
    eta = args.eta
    budget_type = args.budget_type
evaluation = args.evaluation
if evaluation == "CV":
    cv = args.cv
searchspace = args.searchspace
iterative_fit = args.iterative_fit == "True"
early_stopping = args.early_stopping == "True"
posthoc_ensembles = args.posthoc_ensembles

tempdir = tempfile.mkdtemp()
autosklearn_directory = os.path.join(tempdir, 'dir')

# Resampling strategies
if evaluation == "holdout" and iterative_fit:
    resampling_strategy_arguments = {}
    resampling_strategy = 'holdout-iterative-fit'
    resampling_strategy_string = '%s' % resampling_strategy
elif evaluation == "holdout" and not iterative_fit:
    resampling_strategy_arguments = {}
    resampling_strategy = 'holdout'
    resampling_strategy_string = '%s' % resampling_strategy
elif evaluation == "CV" and not iterative_fit:
    N_FOLDS = cv
    resampling_strategy = 'cv'
    resampling_strategy_arguments = {'folds': N_FOLDS}
    resampling_strategy_string = '%s:%d' % (resampling_strategy, N_FOLDS)
elif evaluation == "CV" and iterative_fit:
    N_FOLDS = cv
    resampling_strategy = 'cv-iterative-fit'
    resampling_strategy_arguments = {'folds': N_FOLDS}
    resampling_strategy_string = '%s:%d' % (resampling_strategy, N_FOLDS)
else:
    raise ValueError("Unknown resampling strategy")

configuration_output_dir = os.path.join(working_directory, "%s" % model)
try:
    os.makedirs(configuration_output_dir)
except:
    pass

fid_string = "%s" % fidelity
if fidelity == "SH":
    fid_string = "%s-eta%d-%s" % (fidelity, eta, budget_type[0])

eva_string = "%s" % evaluation
if evaluation == "CV":
    eva_string = "%d%s" % (cv, evaluation)

if overwrite_output_dir_name is None:
    #RF_None_holdout_iterative_if_es_<taskid>_<seed>_<warm>_<ens>
    dir_name = "_".join(["%s" % i for i in [model, fid_string, eva_string, searchspace,
                                            "es" if early_stopping else "nes",
                                            "if" if iterative_fit else "nif",
                                            task_id, seed, initial_configurations_via_metalearning, 0]
                         ])
else:
    dir_name = overwrite_output_dir_name + ('_%d_%d_None_0' % (task_id, seed))
tmp_dir = os.path.join(configuration_output_dir, dir_name)

if os.path.exists(tmp_dir):
    print('Output directory %s already exists - no need to run this again!' % tmp_dir)
    exit(0)
else:
    os.makedirs(tmp_dir)
    software_stats_file = os.path.join(tmp_dir, 'software.txt')
    with open(software_stats_file, 'wt') as fh:
        fh.write(strio.read())
    argparser_content  = {}
    for key, value in vars(args).items():
        argparser_content[key] = value
    argparser_content_file = os.path.join(tmp_dir, 'arguments.json')
    with open(argparser_content_file, 'wt') as fh:
        json.dump(argparser_content, fh, indent=4)

X_train, y_train, X_test, y_test, cat = load_task(task_id)

iterative_wo_early_stopping = ['extra_trees', 'PassiveAggressiveWOEarlyStopping', 'random_forest',
                               'SGDWOEarlyStopping', 'GradientBoostingClassifierWOEarlyStopping']
iterative_w_early_stopping = ['extra_trees', 'passive_aggressive', 'random_forest', 'sgd', 'gradient_boosting']

if not early_stopping:
    add_classifier_wo_early_stopping()

if searchspace == "iterative":
    include_estimator = iterative_w_early_stopping if early_stopping else iterative_wo_early_stopping
    include_preprocessor = ["no_preprocessing", ]
elif searchspace == "iterative-preproc":
    include_estimator = iterative_w_early_stopping if early_stopping else iterative_wo_early_stopping
    include_preprocessor = None
elif searchspace == "full":
    assert early_stopping is True
    include_estimator = None
    include_preprocessor = None
else:
    raise ValueError(searchspace)

initial_budget = False
if fidelity == "SH":
    if eta == 3:
        initial_budget = 10.0
    elif eta == 4:
        initial_budget = 5.0

if model == "RF":
    if fidelity is None:
        get_smac_object_callback = get_smac_object_callback(portfolio)
    elif fidelity == "SH":
        get_smac_object_callback = \
            get_sh_or_hb_object_callback(budget_type, "sh", eta, initial_budget, portfolio)
elif model == None:
    if fidelity is None:
        get_smac_object_callback = get_random_search_object_callback
    elif fidelity == "SH":
        get_smac_object_callback = \
            get_random_search_for_sh_or_hb_object_callback(budget_type, "sh", eta, initial_budget)
else:
    raise ValueError()


automl = AutoSklearnClassifier(
    time_left_for_this_task=time_limit,
    per_run_time_limit=per_run_time_limit,
    initial_configurations_via_metalearning=initial_configurations_via_metalearning,
    ensemble_size=0,
    ensemble_nbest=0,
    seed=seed,
    disable_evaluator_output=['model'],
    ml_memory_limit=memory_limit,
    resampling_strategy=resampling_strategy,
    resampling_strategy_arguments=resampling_strategy_arguments,
    delete_tmp_folder_after_terminate=False,
    max_models_on_disc=None,
    tmp_folder=autosklearn_directory,
    include_estimators=include_estimator,
    include_preprocessors=include_preprocessor,
    get_smac_object_callback=get_smac_object_callback,
    metadata_directory=metadata_directory,
)

trajectory = [(0.0, 1.0)]
min_cost = np.inf
loss = np.inf

crashed = False
try:
    automl.fit(
        X_train, y_train,
        dataset_name=str(task_id),
        feat_type=cat,
        metric=balanced_accuracy,
        X_test=X_test,
        y_test=y_test,
    )
    print('Finished Auto-sklearn fitting', flush=True)

    with open(os.path.join(autosklearn_directory, '.auto-sklearn', 'start_time_%d' % seed)) as fh:
        start_time = float(fh.read())

    for run_key, run_data in automl._automl[0].runhistory_.data.items():
        if run_data.cost < min_cost:
            if 'test_loss' in run_data.additional_info:
                num_run = run_data.additional_info['num_run']
                prediction_file_path = os.path.join(
                    autosklearn_directory,
                    '.auto-sklearn',
                    'predictions_ensemble',
                    'predictions_ensemble_%d_%d_*.npy' % (seed, num_run),
                )
                prediction_file_path = glob.glob(prediction_file_path)[0]
                mtime = os.path.getmtime(prediction_file_path)
                loss = run_data.additional_info['test_loss']
                min_cost = run_data.cost
                trajectory.append((mtime - start_time, loss))
    print(trajectory)

except Exception as e:
    print('Running Auto-sklearn failed due to:')
    import traceback
    print(traceback.print_exc())
    crashed = True
    pass
    #new_autosklearn_path = os.path.join(tmp_dir, 'auto-sklearn-output')
    #shutil.copytree(autosklearn_directory, new_autosklearn_path)
    #try:
    #    shutil.rmtree(autosklearn_directory)
    #except:
    #    pass
    #raise e

# Store searchspace for later examination if run not crashed
if not crashed:
    cs = automl._automl[0].configuration_space
    with open(os.path.join(tmp_dir, 'space.json'), 'w') as fh:
        fh.write(write(cs))

result = dict()
result[0] = {
    'task_id': task_id,
    'time_limit': time_limit,
    'loss': loss,
    'trajectory': trajectory
}

time_stamp_dict = {}
for dirpath, dirnames, filenames in os.walk(autosklearn_directory, topdown=False):
    time_stamp_dict[dirpath] = {}
    for filename in filenames:
        time_stamp_dict[dirpath][filename] = os.path.getmtime(os.path.join(dirpath, filename))
# Save timestamps, so we can compute ensemble performance over time
with open(os.path.join(autosklearn_directory, "timestamps.json"), "w") as fh:
    json.dump(time_stamp_dict, fh, indent=4)

with open(os.path.join(autosklearn_directory, 'result.json'), 'wt') as fh:
    json.dump(result, fh, indent=4)

if posthoc_ensembles:
    ensembles.run_ensemble_builder.main(
        task_id=task_id,
        ensemble_dir=autosklearn_directory,
        performance_range_threshold=0.0,
        ensemble_size=50,
        max_keep_best=1.0,
        seed=seed,
        only_portfolio_runs=False,
        call_from_cmd=False,
    )

for dirpath, dirnames, filenames in os.walk(autosklearn_directory, topdown=False):
    time_stamp_dict[dirpath] = {}
    print(dirpath, dirnames, filenames)
    for filename in filenames:
        print(filename, filename.startswith('ensemble_results'))
        if crashed and ".log" in filename:
            continue
        elif filename.startswith('ensemble_results'):
            continue
        elif filename in ['traj_aclib2.json', 'trajectory.json', 'stats.json', 'runhistory.json']:
            continue
        elif filename.startswith('start_time') and keep_predictions:
            continue
        elif filename == 'true_targets_ensemble.npy' and keep_predictions:
            continue
        elif keep_predictions and filename == "timestamps.json":
            continue
        elif filename.startswith('predictions_ensemble') or filename.startswith('predictions_test'):
            filepath = os.path.join(dirpath, filename)
            if keep_predictions:
                target_filepath = filepath + '.gz'
                try:
                    with open(filepath, 'rb') as fh:
                        tmp = pickle.load(fh)
                        with gzip.open(target_filepath, 'wb') as fh:
                            pickle.dump(tmp, fh)
                except:
                    # If the file cannot be opened, we keep it!
                    continue
            os.remove(filepath)
        else:
            os.remove(os.path.join(dirpath, filename))
    for dirname in dirnames:
        if dirname.startswith('run_'):
            continue
        elif dirname in ['smac3-output']:
            continue
        elif dirname in ['.auto-sklearn', 'predictions_ensemble', 'predictions_test'] and keep_predictions:
            continue
        else:
            os.rmdir(os.path.join(dirpath, dirname))

with open(os.path.join(tmp_dir, 'result.json'), 'wt') as fh:
    json.dump(result, fh, indent=4)
new_autosklearn_path = os.path.join(tmp_dir, 'auto-sklearn-output')
shutil.copytree(autosklearn_directory, new_autosklearn_path)
if posthoc_ensembles:
    shutil.move(
        os.path.join(new_autosklearn_path, 'ensemble_results_0.000000thresh_50size_1.000000best'),
        os.path.join(new_autosklearn_path, '..', 'ensemble_results_0.000000thresh_50size_1.000000best'),
    )
try:
    shutil.rmtree(autosklearn_directory)
except:
    pass
