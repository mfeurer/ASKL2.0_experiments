import argparse
import copy
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import sklearn.model_selection

sys.path.append('.')
from utils import RF, RFSH, IMP0, \
    ASKL_FULL, ASKL_FULL_RANDOM, ASKL_ITER, ASKL_ITER_RANDOM
from utils import dataset_dc, method_dc, automl_metadata, openml_automl_benchmark

NSEEDS = 10
tl_settings = {
    "10MIN": {
        "prtl": 60,
        "tl": 600,
        "memory": 4000,
    },
    "20MIN": {
        "prtl": 120,
        "tl": 1200,
        "memory": 4000,
    },
    "60MIN": {
        "prtl": 360,
        "tl": 3600,
        "memory": 4000,
    },
    "10H": {
        "prtl": 3600,
        "tl": 36000,
        "memory": 4000,
    }
}

def write_cmd(cmd_list, cmd_file):
    if len(cmd_list) < 10000:
        with open(cmd_file, "w") as fh:
            fh.write("\n".join(cmd_list))
        print("Written %d commands to %s" % (len(cmd_list), cmd_file))
    else:
        print("Found more than 10000 cmds (%d)" % len(cmd_list))
        ct = 0
        while ct < len(cmd_list):
            ct += 10000
            with open(cmd_file + "_%d" % ct, "w") as fh:
                fh.write("\n".join(cmd_list[ct-10000:ct]))
            print("Written %d commands to %s" % (len(cmd_list[ct-10000:ct]), cmd_file))


def run_autosklearn(taskset, setting, methods, working_directory, nseeds, keep_predictions=False,
                    initial_configurations_via_metalearning=0, metadata_directory=None,
                    autobild_ensembles=False):
    # Creates cmd file to run autosklearn
    cmd_tmpl = "python ./run_autosklearn/run_autosklearn.py --time-limit %d --memory-limit %d " \
               "--per-run-time-limit %d --working-directory %s " \
               "--initial-configurations-via-metalearning %d --task-id %d -s %d"

    if autobild_ensembles:
        cmd_tmpl += " --posthoc-ensemble"

    cmd_list = []
    for seed in range(nseeds):
        for tid in dataset_dc[taskset]:
            for set in methods:
                for method in set:
                    cmd_base = cmd_tmpl % (tl_settings[setting]["tl"],
                                           tl_settings[setting]["memory"],
                                           tl_settings[setting]["prtl"],
                                           working_directory,
                                           initial_configurations_via_metalearning,
                                           tid,
                                           seed)
                    for k in method_dc[method]:
                        cmd_base += " --%s %s" % (k, method_dc[method][k])
                    if keep_predictions:
                        cmd_base += " --keep-predictions"
                    if metadata_directory:
                        cmd_base += (" --metadata-directory %s" % metadata_directory)
                    cmd_list.append(cmd_base)

    if os.path.isdir(working_directory):
        print("Working directory %s already exists: Abort!" % working_directory)
        sys.exit(1)
    else:
        os.makedirs(working_directory)

    cmd_file = os.path.join(working_directory, "%s_cmds.txt" % setting)
    write_cmd(cmd_list, cmd_file)


def run_get_portfolio_configurations(input_dir, output_dir, methods, nseeds, taskset, setting):
    cmd_tmpl = "python portfolio_matrix/get_portfolio_configurations.py --input_dir %s " \
               "--method %s --output_dir %s --nseeds %d --taskset %s"
    server_cmd_tmpl = 'python portfolio_matrix/server.py --input-dir %s --port 1234%d --host kisbat2'
    worker_cmd_tmpl = 'python portfolio_matrix/worker.py --input-dir %s --memory-limit %s ' \
                      '--time-limit 10000 --per-run-time-limit %s --working-directory %s/matrix/ ' \
                      '--host kisbat2 --port 1234%d'

    cmd_list = []
    server_cmd_list = []
    worker_cmd_dc = {}
    for set in methods:
        for idx, method in enumerate(set):
            cmd = cmd_tmpl % (input_dir, method, output_dir, nseeds, taskset)
            cmd_list.append(cmd)

            # build server command
            cmd = server_cmd_tmpl % (output_dir + "/" + method, idx)
            for k in ["searchspace", "evaluation", "iterative-fit", "early-stopping"]:
                cmd += " --%s %s" % (k, method_dc[method][k])
            if method_dc[method]["evaluation"] == "CV":
                cmd += " --cv %s" % method_dc[method]["cv"]
            cmd += " &"
            server_cmd_list.append(cmd)

            # build worker command
            worker_cmd_dc[method] = []
            cmd = worker_cmd_tmpl % (output_dir + "/" + method, tl_settings[setting]["memory"],
                                     tl_settings[setting]["prtl"], output_dir + "/" + method, idx)
            for i in range(100):
                worker_cmd_dc[method].append(cmd)

    if os.path.isdir(output_dir):
        print("Working directory %s already exists: Abort!" % output_dir)
        sys.exit(1)
    else:
        os.makedirs(output_dir)

    cmd_file = os.path.join(output_dir, "get_portfolio_configurations.cmd")
    with open(cmd_file, "w") as fh:
        fh.write("\n".join(cmd_list))
    print("Written %d commands to %s" % (len(cmd_list), cmd_file))

    cmd_file = os.path.join(output_dir, "server.cmd")
    with open(cmd_file, "w") as fh:
        fh.write("\n".join(server_cmd_list))
    print("Written %d commands to %s" % (len(server_cmd_list), cmd_file))

    for m in worker_cmd_dc:
        cmd_file = os.path.join(output_dir, "worker_%s.txt" % m)
        with open(cmd_file, "w") as fh:
            fh.write("\n".join(worker_cmd_dc[m]))
        print("Written %d commands to %s" % (len(worker_cmd_dc[m]), cmd_file))


def run_create_matrix(input_dir, output_dir, methods):
    cmd_tmpl = "python portfolio_matrix/create_matrix.py --working-directory %s/%s/matrix/ --save-to %s/%s/"
    cmds = []
    for set in methods:
        for method in set:
            cmd = cmd_tmpl % (input_dir, method, input_dir, method)
            cmds.append(cmd)
    cmd_file = os.path.join(output_dir, "build_matrix.cmd")
    with open(cmd_file, "w") as fh:
        fh.write("\n".join(cmds))
    print("Written %d commands to %s" % (len(cmds), cmd_file))


def run_create_symlinks(input_dir, setting):
    print(os.listdir(input_dir))
    replacements = {'_None_': '_SH-eta4-i_'}
    for dir in os.listdir(input_dir):
        for idx, method in enumerate(RF):
            for target, replace in replacements.items():
                if method == dir:
                    target_dir = os.path.join(input_dir, dir.replace(target, replace))
                    print("Create symlink from %s to %s" % (dir, target_dir))
                    os.symlink(dir, target_dir)


def run_create_portfolio(input_dir, output_dir, nseeds, portfolio_size, methods, setting):
    call_template = (
        'python portfolio/build_portfolio.py '
        '--input-directory %s '
        '--portfolio-size %d '
    )

    commands = []
    commands_cv = []
    execute_portfolio_calls = []

    os.makedirs(output_dir, exist_ok=True)

    for set in methods:
        for idx, method in enumerate(set):
            for seed in range(nseeds):
                call = copy.copy(call_template)
                input_dir_ = os.path.join(input_dir, method)
                output_dir_ = os.path.join(output_dir, method)
                os.makedirs(output_dir_, exist_ok=True)
                portfolio_execution_output_directory = os.path.join(output_dir_, 'portfolio_execution')
                os.makedirs(portfolio_execution_output_directory, exist_ok=True)
                fidelities = method_dc[method]['fidelity']
                for max_runtime in (None, tl_settings[setting]["tl"]):
                    task_to_portfolio = {}

                    if max_runtime is not None:
                        call = call + (' --max-runtime %d' % max_runtime)
                    call = call + (' --fidelities %s' % fidelities)
                    call = call + (' --seed %d' % seed)

                    output_file = os.path.join(
                        output_dir_,
                        '%d_%s_%s_%d.json' % (portfolio_size, fidelities, max_runtime, seed)
                    )

                    main_call = call + (' --output-file %s' % output_file)
                    main_call = main_call % (input_dir_, portfolio_size)
                    commands.append(main_call)
                    metadata_ids = np.array(automl_metadata)

                    kfold = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=seed)
                    for split_ixd, (training_indices, test_indices) in enumerate(
                            kfold.split(metadata_ids)):
                        training_task_ids = metadata_ids[training_indices]
                        test_task_ids = metadata_ids[test_indices]

                        output_file = os.path.join(
                            output_dir_,
                            '%d_%s_%s_%d_%d.json' % (
                            portfolio_size, fidelities, max_runtime, seed, split_ixd)
                        )

                        tmp_call = call + ' --training-task-ids '
                        tmp_call = tmp_call + ' '.join([str(tti) for tti in training_task_ids])
                        tmp_call = tmp_call + ' --output-file %s' % output_file
                        tmp_call = tmp_call % (input_dir_, portfolio_size)
                        commands_cv.append(tmp_call)

                        for task_id in test_task_ids:

                            task_to_portfolio[int(task_id)] = output_file

                            portfolio_execution_output_file = os.path.join(
                                portfolio_execution_output_directory,
                                '%d_%s_%s_%d_%d.json' % (
                                portfolio_size, fidelities, max_runtime, seed, task_id)
                            )

                            execute_portfolio_call = (
                                'python portfolio/execute_portfolio.py '
                                '--portfolio-file %s '
                                '--input-directory %s '
                                '--output-file %s '
                                '--fidelities %s '
                                '--task-id %d '
                                '--max-runtime %d'
                            ) % (output_file, input_dir_,
                                portfolio_execution_output_file,
                                fidelities, int(task_id),
                                tl_settings[setting]["tl"]
                            )
                            execute_portfolio_calls.append(execute_portfolio_call)

                    if max_runtime:
                        task_to_portfolio_filename = os.path.join(
                            output_dir_,
                            'task_to_portfolio_max_runtime_%d.json' % seed,
                        )
                    else:
                        task_to_portfolio_filename = os.path.join(
                            output_dir_,
                            'task_to_portfolio_%d.json' % seed,
                        )
                    with open(task_to_portfolio_filename, 'wt') as fh:
                        json.dump(task_to_portfolio, fh, indent=4)

    call_file = os.path.join(output_dir, 'build_portfolio.cmd')
    write_cmd(commands, call_file)

    cv_call_file = os.path.join(output_dir, 'build_portfolio_cv.cmd')
    write_cmd(commands_cv, cv_call_file)

    execute_portfolio_call_file = os.path.join(output_dir, 'execute_portfolio_cv.cmd')
    write_cmd(execute_portfolio_calls, execute_portfolio_call_file)


def run_autosklearn_with_portfolio(taskset, setting, methods, working_directory, nseeds,
                                   portfolio_directory, keep_predictions=False,
                                   autobild_ensembles=False,
                                   portfolio_from_dictionary_file=False):
    # Creates cmd file to run autosklearn
    cmd_tmpl = "python ./run_autosklearn/run_autosklearn.py --time-limit %d --memory-limit %d " \
               "--working-directory %s " \
               "--initial-configurations-via-metalearning %d --task-id %d -s %d"

    if autobild_ensembles:
        cmd_tmpl += " --posthoc-ensemble"

    learned_cmd_list = []
    fixed_cmd_list = []
    for seed in range(nseeds):
        for tid in dataset_dc[taskset]:
            for set in methods:
                for method in set:
                    for prtl, portfolio_tl in (
                        (None, tl_settings[setting]["tl"]),
                        (tl_settings[setting]["prtl"], 'None'),
                    ):
                        dir = os.path.join(working_directory, str(prtl))
                        cmd_base = cmd_tmpl % (tl_settings[setting]["tl"],
                                               tl_settings[setting]["memory"],
                                               dir,
                                               0,
                                               tid,
                                               seed)
                        for k in method_dc[method]:
                            cmd_base += " --%s %s" % (k, method_dc[method][k])
                        if prtl:
                            cmd_base += " --per-run-time-limit %d" % prtl
                        if keep_predictions:
                            cmd_base += " --keep-predictions"
                        fidelities = method_dc[method]['fidelity']

                        if portfolio_from_dictionary_file:
                            dictionary_file = os.path.join(
                                os.path.abspath(portfolio_directory),
                                method,
                                ('task_to_portfolio_max_runtime_%d.json' % seed)
                                if portfolio_tl != 'None'
                                else ('task_to_portfolio_%d.json' % seed)
                            )
                            with open(dictionary_file) as fh:
                                portfolio_dict = json.load(fh)
                            portfolio_file_name = portfolio_dict[str(tid)]
                        else:
                            portfolio_file_name = os.path.join(
                                os.path.abspath(portfolio_directory),
                                method,
                                "32_%s_%s_%d.json" % (fidelities, portfolio_tl, seed),
                            )
                        cmd_base = cmd_base + " --portfolio-file " + portfolio_file_name
                        if prtl:
                            fixed_cmd_list.append(cmd_base)
                        else:
                            learned_cmd_list.append(cmd_base)

    if os.path.isdir(working_directory):
        print("Working directory %s already exists: Abort!" % working_directory)
        sys.exit(1)
    else:
        os.makedirs(working_directory)

    cmd_file = os.path.join(working_directory, "learned_%s_cmds.txt" % setting)
    write_cmd(learned_cmd_list, cmd_file)
    cmd_file = os.path.join(working_directory, "fixed_%s_cmds.txt" % setting)
    write_cmd(fixed_cmd_list, cmd_file)
    cmd_file = os.path.join(working_directory, "%s_cmds.txt" % setting)
    write_cmd(learned_cmd_list + fixed_cmd_list, cmd_file)


def run_AutoAuto_build(methods, portfolio_dir, matrices_dir, output_dir, nseeds,
                       metadata_type='portfolio'):
    if os.path.isdir(output_dir):
        print("Working directory %s already exists: Abort!" % output_dir)
        sys.exit(1)
    else:
        os.makedirs(output_dir)
    metafeatures_directory = os.path.join(output_dir, 'metafeatures')
    os.makedirs(metafeatures_directory)

    commands = []

    call_template = 'python ./autoauto/run_autoauto.py'

    for selector_type in ('static', 'dynamic'):
        for pwmrt in (True, False):
            call = call_template + (' --type %s' % selector_type)
            call = call + (' --metadata-type %s' % metadata_type)
            call = call + (' --metafeatures-directory %s' % metafeatures_directory)
            if pwmrt:
                call = call + ' --portfolios-with-max-runtime'

            if metadata_type == 'portfolio':
                call = call + ' --input-directories'
                for set in methods:
                    for method in set:
                        inp_dir = os.path.join(portfolio_dir, method)
                        call = call + ' ' + inp_dir
            elif metadata_type in ['full_runs', 'full_runs_ensemble']:
                call = call + (' --input-directories %s' % portfolio_dir)
                call = call + ' --only'
                for set in methods:
                    for method in set:
                        call = call + (' %s' % method)
            else:
                raise ValueError(metadata_type)
            call = call + (' --matrices-directory %s' % matrices_dir)

            for seed in range(nseeds):
                call_with_seed = call + (' --seed %d' % seed)
                output_file = os.path.join(output_dir, selector_type,
                                           'pwmrt' if pwmrt else 'pWOmrt', '%d.pkl' % seed)
                call_with_seed = call_with_seed + (' --output-file %s' % output_file)
                commands.append(call_with_seed)

    cmd_file = os.path.join(output_dir, "commands.txt")
    write_cmd(cmd_list=commands, cmd_file=cmd_file)


def run_AutoAuto_simulate(selector_dir, run_with_portfolio_dir, output_dir, nseeds, setting,
                          add_symlinks=False, add_symlinks_and_stats_file=False,
                          add_no_fallback=False):

    if add_symlinks:
        assert os.path.isdir(output_dir)
    else:
        if os.path.isdir(output_dir):
            print("Working directory %s already exists: Skip!" % output_dir)
            # sys.exit(1)
        else:
            os.makedirs(output_dir)

    learned_cmd_list = []
    fixed_cmd_list = []
    call_template = 'python run_autosklearn/simulate_autoautoml.py'

    selector_mappinf = {
        'static': 'static',
        'dynamic': 'dynamic',
        'dynamic-no-fallback': 'dynamic',
    }
    for selector_type in ('static', 'dynamic', 'dynamic-no-fallback'):

        if not add_no_fallback and selector_type == 'dynamic-no-fallback':
            continue

        for pwmrt in (True, False):

            output_dir_ = os.path.join(output_dir, selector_type)

            if pwmrt:
                subdir = str(tl_settings[setting]['prtl'])
            else:
                subdir = 'None'

            input_dir = os.path.join(run_with_portfolio_dir, subdir)
            output_dir_ = os.path.join(output_dir_, subdir)

            input_dir = os.path.join(input_dir, 'RF')

            for seed in range(nseeds):
                for task_id in openml_automl_benchmark:
                    selector_file = os.path.join(
                        selector_dir,
                        selector_mappinf[selector_type],
                        'pwmrt' if pwmrt else 'pWOmrt',
                        '%d.pkl' % seed
                    )
                    call = call_template + (' --selector-file %s' % selector_file)
                    call = call + (' --task-id %d' % task_id)
                    call = call + (' --seed %d' % seed)
                    call = call + (' --input-dir %s' % input_dir)
                    call = call + (' --output-dir %s' % output_dir_)
                    call = call + (' --max-runtime-limit %s' % subdir)
                    if add_symlinks:
                        call = call + (' --create-symlink --only-check-stats-file')
                    elif add_symlinks_and_stats_file:
                        call = call + (' --create-symlink')
                    if selector_type == 'dynamic-no-fallback':
                        call = call + (' --disable-fallback')
                    if pwmrt:
                        fixed_cmd_list.append(call)
                    else:
                        learned_cmd_list.append(call)

    if not add_symlinks:
        cmd_file = os.path.join(output_dir, "autoauto_simulation_commands.txt")
        learned_cmd_file = os.path.join(output_dir, "learned_autoauto_simulation_commands.txt")
        fixed_cmd_file = os.path.join(output_dir, "fixed_autoauto_simulation_commands.txt")
    else:
        cmd_file = os.path.join(output_dir, "autoauto_simulation_commands_symlinks.txt")
        learned_cmd_file = os.path.join(output_dir, "learned_autoauto_simulation_commands_symlinks.txt")
        fixed_cmd_file = os.path.join(output_dir, "fixed_autoauto_simulation_commands_symlinks.txt")

    write_cmd(learned_cmd_list + fixed_cmd_list, cmd_file)
    write_cmd(learned_cmd_list, learned_cmd_file)
    write_cmd(fixed_cmd_list, fixed_cmd_file)


def prune_run_with_portfolio(setting, commands_dir, autoauto_dir, rq_prefix):
    for prefix, cmd_file in [
        ('', os.path.join(commands_dir, '%s_cmds.txt' % setting)),
        ('fixed_', os.path.join(commands_dir, 'fixed_%s_cmds.txt' % setting)),
        ('learned_', os.path.join(commands_dir, 'learned_%s_cmds.txt' % setting)),
    ]:
        new_commands = []
        if not os.path.exists(cmd_file):
            continue
        with open(cmd_file) as fh:
            commands = fh.read().split('\n')
        glob_cmd = os.path.join(autoauto_dir, '*', '*', '*.json')
        globs = glob.glob(glob_cmd)
        print(glob_cmd, len(globs))
        for entry in globs:
            with open(entry) as fh:
                jason = json.load(fh)
            try:
                per_run_time_limit = jason['max_runtime_limit']
            except KeyError:
                continue
            if per_run_time_limit == 'None':
                runtime_limit = tl_settings[setting]['tl']
            else:
                runtime_limit = 'None'
            chosen_method = jason['chosen_method']
            seed = jason['seed']
            task_id = jason['task_id']
            portfolio_file_string = os.path.join(setting, 'ASKL_create_portfolio', chosen_method)
            fidelities = 'SH' if 'SH' in chosen_method else 'None'
            portfolio_file_string = os.path.join(
                portfolio_file_string,
                '32_%s_%s_%d.json' % (fidelities, runtime_limit, seed)
            )
            for command in commands:
                # The weird construct of the portfolio file string removes the time horizon indicator
                if '/'.join(portfolio_file_string.split('/')[1:]) in command and ('--task-id %d' % task_id) in command:
                    new_commands.append(command)

        new_commands_file = os.path.join(
            commands_dir,
            '%s%s%s_cmds_selected.txt' % (rq_prefix, prefix, setting),
        )
        write_cmd(new_commands, new_commands_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--do',
                        choices=("ASKL_metadata", "ASKL_metadata_full",
                                 "ASKL_metadata_full_run_with_portfolio",
                                 "ASKL_improvement_zero",
                                 "ASKL_automldata", "ASKL_automldata_w_ensemble",
                                 "ASKL_automldata_w_ensemble_w_knd",
                                 "ASKL_automldata_baseline_iter",
                                 "ASKL_automldata_baseline_iter_no_metalearning",
                                 "ASKL_automldata_baseline_iter_random",
                                 "ASKL_automldata_baseline_full",
                                 "ASKL_automldata_baseline_full_no_metalearning",
                                 "ASKL_automldata_baseline_full_random",
                                 "ASKL_getportfolio", "run_create_matrix", "run_create_symlinks",
                                 "ASKL_create_portfolio",
                                 "ASKL_run_with_portfolio", "ASKL_run_with_portfolio_w_ensemble",
                                 "AutoAuto_build", "AutoAuto_simulate",
                                 "AutoAuto_simulate_create_posthoc_symlinks",
                                 "prune_run_with_portfolio",
                                 "AutoAuto_build_full_data", "AutoAuto_simulate_full_data",
                                 "RQ1_AutoAuto_build", "RQ1_AutoAuto_simulate",
                                 "RQ1_AutoAuto_simulate_create_posthoc_symlinks",
                                 "RQ1_prune_run_with_portfolio",
                                 "RQ2.1_AutoAuto_build", "RQ2.1_AutoAuto_simulate",
                                 "RQ2.2_AutoAuto_build", "RQ2.2_AutoAuto_simulate",
                                 "RQ2.3_AutoAuto_build", "RQ2.3_AutoAuto_simulate",
                                 "RQ3.1_ASKL_run_with_portfolio_w_ensemble",
                                 "RQ3.1_AutoAuto_simulate",
                                 "RQ3.1_prune_run_with_portfolio"),
                        required=True)
    parser.add_argument('--setting', choices=list(tl_settings.keys()), required=True)
    args = parser.parse_args()

    dir_lookup = {
        '10MIN': "60",
        '20MIN': "120",
        '60MIN': "360",
        "10H": "3600",
    }

    if args.do == "ASKL_metadata":
        working_directory = os.path.join(args.setting, args.do)
        run_autosklearn(taskset="automl_metadata", setting=args.setting, methods=(RF, ),
                        working_directory=working_directory, nseeds=3)
    elif args.do == "ASKL_improvement_zero":
        working_directory = os.path.join(args.setting, args.do)
        run_autosklearn(taskset="openml_automl_benchmark", setting=args.setting, methods=(IMP0, ),
                        working_directory=working_directory, nseeds=10)
    elif args.do == "ASKL_metadata_full":
        working_directory = os.path.join(args.setting, args.do)
        run_autosklearn(taskset="automl_metadata", setting=args.setting, methods=(RF, RFSH),
                        working_directory=working_directory, nseeds=10,
                        autobild_ensembles=True)
    elif args.do == "ASKL_metadata_full_run_with_portfolio":
        working_directory = os.path.join(args.setting, args.do)
        portfolio_directory = os.path.join(args.setting, "ASKL_create_portfolio")
        run_autosklearn_with_portfolio(taskset="automl_metadata", setting=args.setting,
                                       methods=(RF, RFSH),
                                       working_directory=working_directory, nseeds=10,
                                       portfolio_directory=portfolio_directory,
                                       autobild_ensembles=True,
                                       portfolio_from_dictionary_file=True)
    elif args.do == "ASKL_automldata":
        working_directory = os.path.join(args.setting, args.do)
        run_autosklearn(taskset="openml_automl_benchmark", setting=args.setting, methods=(RF, RFSH),
                        working_directory=working_directory, nseeds=10)
    elif args.do == "ASKL_automldata_w_ensemble":
        working_directory = os.path.join(args.setting, args.do)
        run_autosklearn(taskset="openml_automl_benchmark", setting=args.setting, methods=(RF, RFSH),
                        working_directory=working_directory, nseeds=10, keep_predictions=True)
    elif args.do == "ASKL_automldata_w_ensemble_w_knd":
        working_directory = os.path.join(args.setting, args.do)
        run_autosklearn(taskset="openml_automl_benchmark", setting=args.setting, methods=(RF, RFSH),
                        working_directory=working_directory, nseeds=10,
                        initial_configurations_via_metalearning=25,
                        metadata_directory='metadata/files/')
    elif args.do == "ASKL_automldata_baseline_iter":
        working_directory = os.path.join(args.setting, args.do)
        run_autosklearn(taskset="openml_automl_benchmark", setting=args.setting,
                        methods=(ASKL_ITER, ), working_directory=working_directory, nseeds=10,
                        keep_predictions=True, initial_configurations_via_metalearning=25,
                        metadata_directory='metadata/files')
    elif args.do == "ASKL_automldata_baseline_iter_no_metalearning":
        working_directory = os.path.join(args.setting, args.do)
        run_autosklearn(taskset="openml_automl_benchmark", setting=args.setting,
                        methods=(ASKL_ITER, ), working_directory=working_directory, nseeds=10,
                        keep_predictions=True, initial_configurations_via_metalearning=0)
    elif args.do == "ASKL_automldata_baseline_iter_random":
        working_directory = os.path.join(args.setting, args.do)
        run_autosklearn(taskset="openml_automl_benchmark", setting=args.setting,
                        methods=(ASKL_ITER_RANDOM, ), working_directory=working_directory,
                        nseeds=10, keep_predictions=True)
    elif args.do == "ASKL_automldata_baseline_full":
        working_directory = os.path.join(args.setting, args.do)
        run_autosklearn(taskset="openml_automl_benchmark", setting=args.setting,
                        methods=(ASKL_FULL, ), working_directory=working_directory, nseeds=10,
                        keep_predictions=True, initial_configurations_via_metalearning=25)
    elif args.do == "ASKL_automldata_baseline_full_no_metalearning":
        working_directory = os.path.join(args.setting, args.do)
        run_autosklearn(taskset="openml_automl_benchmark", setting=args.setting,
                        methods=(ASKL_FULL, ), working_directory=working_directory, nseeds=10,
                        keep_predictions=True, initial_configurations_via_metalearning=0)
    elif args.do == "ASKL_automldata_baseline_full_random":
        working_directory = os.path.join(args.setting, args.do)
        run_autosklearn(taskset="openml_automl_benchmark", setting=args.setting,
                        methods=(ASKL_FULL_RANDOM, ), working_directory=working_directory,
                        nseeds=10, keep_predictions=True)
    elif args.do == "ASKL_getportfolio":
        input_dir = os.path.join(args.setting, "ASKL_metadata")
        output_dir = os.path.join(args.setting, args.do)
        run_get_portfolio_configurations(input_dir=input_dir, output_dir=output_dir,
                                         methods=(RF, ), nseeds=3, taskset="automl_metadata",
                                         setting=args.setting)
    elif args.do == "run_create_matrix":
        input_dir = os.path.join(args.setting, 'ASKL_getportfolio')
        output_dir = os.path.join(args.setting, 'ASKL_getportfolio')
        run_create_matrix(input_dir=input_dir, output_dir=output_dir, methods=(RF, ))
    elif args.do == "run_create_symlinks":
        input_dir = os.path.join(args.setting, 'ASKL_getportfolio')
        run_create_symlinks(input_dir, setting=args.setting)
    elif args.do == "ASKL_create_portfolio":
        input_dir = os.path.join(args.setting, 'ASKL_getportfolio')
        output_dir = os.path.join(args.setting, args.do)
        run_create_portfolio(input_dir=input_dir, output_dir=output_dir, nseeds=10,
                             portfolio_size=32, methods=(RF, RFSH), setting=args.setting)
    elif args.do == "ASKL_run_with_portfolio":
        working_directory = os.path.join(args.setting, args.do)
        portfolio_directory = os.path.join(args.setting, "ASKL_create_portfolio")
        run_autosklearn_with_portfolio(taskset="openml_automl_benchmark", setting=args.setting, methods=(RF, RFSH),
                                       working_directory=working_directory, nseeds=10,
                                       portfolio_directory=portfolio_directory)
    elif args.do == "ASKL_run_with_portfolio_w_ensemble":
        working_directory = os.path.join(args.setting, args.do)
        portfolio_directory = os.path.join(args.setting, "ASKL_create_portfolio")
        run_autosklearn_with_portfolio(taskset="openml_automl_benchmark", setting=args.setting,
                                       methods=(RF, RFSH), working_directory=working_directory,
                                       nseeds=10, portfolio_directory=portfolio_directory,
                                       keep_predictions=True)
    elif args.do == "AutoAuto_build":
        matrices_dir = os.path.join(args.setting, "ASKL_getportfolio")
        portfolio_dir = os.path.join(args.setting, "ASKL_metadata_full_run_with_portfolio",
                                     dir_lookup[args.setting], "RF")
        output_dir = os.path.join(args.setting, args.do)
        run_AutoAuto_build(methods=(RF, RFSH), nseeds=10, portfolio_dir=portfolio_dir,
                           matrices_dir=matrices_dir, output_dir=output_dir,
                           metadata_type='full_runs')
    elif args.do == "AutoAuto_simulate":
        selector_dir = os.path.join(args.setting, "AutoAuto_build")
        run_with_portfolio_dir = os.path.join(args.setting, "ASKL_run_with_portfolio_w_ensemble",)
        output_dir = os.path.join(args.setting, args.do)
        run_AutoAuto_simulate(selector_dir=selector_dir,
                              run_with_portfolio_dir=run_with_portfolio_dir,
                              output_dir=output_dir, nseeds=10, setting=args.setting,
                              add_symlinks_and_stats_file=True,
                              add_no_fallback=True)
    # elif args.do == "AutoAuto_simulate_create_posthoc_symlinks":
    #     selector_dir = os.path.join(args.setting, "AutoAuto_build")
    #     run_with_portfolio_dir = os.path.join(args.setting, "ASKL_run_with_portfolio")
    #     output_dir = os.path.join(args.setting, 'AutoAuto_simulate')
    #     run_AutoAuto_simulate(selector_dir=selector_dir,
    #                           run_with_portfolio_dir=run_with_portfolio_dir,
    #                           output_dir=output_dir, nseeds=10, setting=args.setting,
    #                           add_symlinks=True)
    # elif args.do == "prune_run_with_portfolio":
    #     commands_dir = os.path.join(args.setting, "ASKL_run_with_portfolio")
    #     autoauto_dir = os.path.join(args.setting, 'AutoAuto_simulate')
    #     prune_run_with_portfolio(setting=args.setting, commands_dir=commands_dir,
    #                              autoauto_dir=autoauto_dir, rq_prefix='')
    elif args.do == "RQ1_AutoAuto_build":
        matrices_dir = os.path.join(args.setting, "ASKL_getportfolio")
        portfolio_dir = os.path.join(args.setting, "ASKL_metadata_full_run_with_portfolio",
                                     dir_lookup[args.setting], "RF")
        output_dir = os.path.join(args.setting, args.do)
        run_AutoAuto_build(methods=(RF, RFSH), nseeds=10, portfolio_dir=portfolio_dir,
                           matrices_dir=matrices_dir, output_dir=output_dir,
                           metadata_type='full_runs_ensemble')
    elif args.do == "RQ1_AutoAuto_simulate":
        selector_dir = os.path.join(args.setting, "RQ1_AutoAuto_build")
        run_with_portfolio_dir = os.path.join(args.setting, "ASKL_run_with_portfolio_w_ensemble")
        output_dir = os.path.join(args.setting, args.do)
        run_AutoAuto_simulate(selector_dir=selector_dir,
                              run_with_portfolio_dir=run_with_portfolio_dir,
                              output_dir=output_dir, nseeds=10, setting=args.setting,
                              add_symlinks_and_stats_file=True,
                              add_no_fallback=True)
    # elif args.do == "RQ1_AutoAuto_simulate_create_posthoc_symlinks":
    #     selector_dir = os.path.join(args.setting, "AutoAuto_build")
    #     run_with_portfolio_dir = os.path.join(args.setting, "ASKL_run_with_portfolio_w_ensemble")
    #     output_dir = os.path.join(args.setting, 'RQ1_AutoAuto_simulate')
    #     run_AutoAuto_simulate(selector_dir=selector_dir,
    #                           run_with_portfolio_dir=run_with_portfolio_dir,
    #                           output_dir=output_dir, nseeds=10, setting=args.setting,
    #                           add_symlinks=True)
    # elif args.do == "RQ1_prune_run_with_portfolio":
    #     commands_dir = os.path.join(args.setting, "ASKL_run_with_portfolio_w_ensemble")
    #     autoauto_dir = os.path.join(args.setting, 'RQ1_AutoAuto_simulate')
    #     prune_run_with_portfolio(setting=args.setting, commands_dir=commands_dir,
    #                              autoauto_dir=autoauto_dir, rq_prefix='RQ1_')
    elif args.do == "RQ2.1_AutoAuto_build":
        matrices_dir = os.path.join(args.setting, "ASKL_getportfolio")
        portfolio_dir = os.path.join(args.setting, "ASKL_metadata_full_run_with_portfolio",
                                     dir_lookup[args.setting], "RF",)
        output_dir = os.path.join(args.setting, args.do)

        # No successive halving
        output_dir_ = os.path.join(output_dir, 'no_sh')
        run_AutoAuto_build(methods=(RF, ), nseeds=10, portfolio_dir=portfolio_dir,
                           matrices_dir=matrices_dir, output_dir=output_dir_,
                           metadata_type='full_runs_ensemble')

        # Only successive halving
        output_dir_ = os.path.join(output_dir, 'only_sh')
        run_AutoAuto_build(methods=(RFSH, ), nseeds=10, portfolio_dir=portfolio_dir,
                           matrices_dir=matrices_dir, output_dir=output_dir_,
                           metadata_type='full_runs_ensemble')

        # No cross-validation
        output_dir_ = os.path.join(output_dir, 'no_cv')
        run_AutoAuto_build(methods=(("RF_None_holdout_iterative_es_if",
                                     "RF_SH-eta4-i_holdout_iterative_es_if"), ),
                           nseeds=10, portfolio_dir=portfolio_dir,
                           matrices_dir=matrices_dir, output_dir=output_dir_,
                           metadata_type='full_runs_ensemble')

        # only cross-validation
        output_dir_ = os.path.join(output_dir, 'only_cv')
        run_AutoAuto_build(methods=(("RF_None_3CV_iterative_es_if",
                                     "RF_None_5CV_iterative_es_if",
                                     "RF_None_10CV_iterative_es_if",
                                     "RF_SH-eta4-i_3CV_iterative_es_if",
                                     "RF_SH-eta4-i_5CV_iterative_es_if",
                                     "RF_SH-eta4-i_10CV_iterative_es_if"),),
                           nseeds=10, portfolio_dir=portfolio_dir,
                           matrices_dir=matrices_dir, output_dir=output_dir_,
                           metadata_type='full_runs_ensemble')
    elif args.do == "RQ2.1_AutoAuto_simulate":
        for subset in ('no_sh', 'only_sh', 'no_cv', 'only_cv'):
            selector_dir = os.path.join(args.setting, "RQ2.1_AutoAuto_build", subset)
            run_with_portfolio_dir = os.path.join(args.setting, "ASKL_run_with_portfolio_w_ensemble")
            output_dir = os.path.join(args.setting, args.do, subset)
            run_AutoAuto_simulate(selector_dir=selector_dir,
                                  run_with_portfolio_dir=run_with_portfolio_dir,
                                  output_dir=output_dir, nseeds=10, setting=args.setting,
                                  add_symlinks_and_stats_file=True)
    elif args.do == "RQ2.2_AutoAuto_build":
        matrices_dir = os.path.join(args.setting, "ASKL_getportfolio")
        portfolio_dir = os.path.join(args.setting, "ASKL_metadata_full", "RF")
        output_dir = os.path.join(args.setting, args.do)
        run_AutoAuto_build(methods=(RF, RFSH), nseeds=10, portfolio_dir=portfolio_dir,
                           matrices_dir=matrices_dir, output_dir=output_dir,
                           metadata_type='full_runs_ensemble')
    elif args.do == "RQ2.2_AutoAuto_simulate":
        selector_dir = os.path.join(args.setting, "RQ2.2_AutoAuto_build")
        run_with_portfolio_dir = os.path.join(args.setting, "ASKL_automldata_w_ensemble")
        output_dir = os.path.join(args.setting, args.do)
        run_AutoAuto_simulate(selector_dir=selector_dir,
                              run_with_portfolio_dir=run_with_portfolio_dir,
                              output_dir=output_dir, nseeds=10, setting=args.setting,
                              add_symlinks_and_stats_file=True)
    elif args.do == "RQ2.3_AutoAuto_build":
        matrices_dir = os.path.join(args.setting, "ASKL_getportfolio")
        portfolio_dir = os.path.join(args.setting, "ASKL_create_portfolio")
        output_dir = os.path.join(args.setting, args.do)
        run_AutoAuto_build(methods=(RF, RFSH), nseeds=10, portfolio_dir=portfolio_dir,
                           matrices_dir=matrices_dir, output_dir=output_dir,
                           metadata_type='portfolio')
    elif args.do == "RQ2.3_AutoAuto_simulate":
        selector_dir = os.path.join(args.setting, "RQ2.3_AutoAuto_build")
        run_with_portfolio_dir = os.path.join(args.setting, "ASKL_run_with_portfolio_w_ensemble")
        output_dir = os.path.join(args.setting, args.do)
        run_AutoAuto_simulate(selector_dir=selector_dir,
                              run_with_portfolio_dir=run_with_portfolio_dir,
                              output_dir=output_dir, nseeds=10, setting=args.setting,
                              add_symlinks_and_stats_file=True,
                              add_no_fallback=True)

    # elif args.do == "RQ3.1_ASKL_run_with_portfolio_w_ensemble":
    #     for setting in tl_settings:
    #         if args.setting == setting:
    #             continue
    #         working_directory = os.path.join(args.setting, args.do, setting)
    #         portfolio_directory = os.path.join(setting, "ASKL_create_portfolio")
    #         run_autosklearn_with_portfolio(taskset="openml_automl_benchmark", setting=args.setting,
    #                                        methods=(RF, RFSH), working_directory=working_directory,
    #                                        nseeds=10, portfolio_directory=portfolio_directory,
    #                                        keep_predictions=True)
    # elif args.do == "RQ3.1_AutoAuto_simulate":
    #     for setting in tl_settings:
    #         if args.setting == setting:
    #             continue
    #         selector_dir = os.path.join(args.setting, "AutoAuto_build")
    #         run_with_portfolio_dir = os.path.join(args.setting,
    #                                               "RQ3.1_ASKL_run_with_portfolio_w_ensemble",
    #                                               setting)
    #         output_dir = os.path.join(args.setting, args.do, setting)
    #         run_AutoAuto_simulate(selector_dir=selector_dir,
    #                               run_with_portfolio_dir=run_with_portfolio_dir,
    #                               output_dir=output_dir, nseeds=10, setting=args.setting)
    # elif args.do == "RQ3.1_prune_run_with_portfolio":
    #     for setting in tl_settings:
    #         if args.setting == setting:
    #             continue
    #         commands_dir = os.path.join(args.setting,
    #                                     "RQ3.1_ASKL_run_with_portfolio_w_ensemble", setting)
    #         autoauto_dir = os.path.join(args.setting, 'RQ3.1_AutoAuto_simulate', setting)
    #         prune_run_with_portfolio(setting=args.setting, commands_dir=commands_dir,
    #                                  autoauto_dir=autoauto_dir, rq_prefix='')
    # elif args.do == "RQ3.1_AutoAuto_simulate_create_posthoc_symlinks":
    #     for setting in tl_settings:
    #         if args.setting == setting:
    #             continue
    #
    #         selector_dir = os.path.join(args.setting, "AutoAuto_build")
    #         run_with_portfolio_dir = os.path.join(args.setting,
    #                                               "RQ3.1_ASKL_run_with_portfolio_w_ensemble",
    #                                               setting)
    #         output_dir = os.path.join(args.setting, 'RQ3.1_AutoAuto_simulate', setting)
    #         run_AutoAuto_simulate(selector_dir=selector_dir,
    #                               run_with_portfolio_dir=run_with_portfolio_dir,
    #                               output_dir=output_dir, nseeds=10, setting=args.setting,
    #                               add_symlinks=True)
    else:
        print("Don't know what to do: Exit!")

