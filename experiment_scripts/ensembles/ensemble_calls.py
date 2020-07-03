import argparse
import glob
import itertools
import os
import sys

this_dir = os.path.dirname(__file__)
main_dir = os.path.abspath(os.path.join(this_dir, '..'))
sys.path.append(main_dir)
from utils import RF, RFSH, openml_automl_benchmark


parser = argparse.ArgumentParser()
parser.add_argument('--time-dir', choices=['10MIN', '20MIN', '60MIN', '10H'], required=True)
args = parser.parse_args()
TIME_DIR = args.time_dir

CMD = "python ensembles/run_ensemble_builder.py"

# baseline ensembles
jobs = [
    (('RF_None_holdout_full_es_nif', ), "ASKL_automldata_baseline_full", 1, False),
    (('RF_None_holdout_full_es_nif', ), "ASKL_automldata_baseline_full_no_metalearning", 1, False),
    (('None_None_holdout_full_es_nif', ), "ASKL_automldata_baseline_full_random", 1, False),
    (('RF_None_holdout_iterative_es_nif', ), "ASKL_automldata_baseline_iter", 1, False),
    (('RF_None_holdout_iterative_es_nif', ), "ASKL_automldata_baseline_iter_no_metalearning", 1, False),
    (('None_None_holdout_iterative_es_nif', ), "ASKL_automldata_baseline_iter_random", 1, False),
    (itertools.chain(RF, RFSH), "ASKL_run_with_portfolio_w_ensemble", 2, True),
    (itertools.chain(RF, RFSH), "ASKL_automldata_w_ensemble", 1, False),
    (itertools.chain(RF, RFSH), "RQ3.1_ASKL_run_with_portfolio_w_ensemble", 3, False),
]

nseeds = 10
ensemble_size = [50]
performance_range_threshold = [0.0]
max_keep_best = [1.0]

for set, wd, n_stars, only_portfolio in jobs:
    for setting in set:
        print(setting, wd)
        outfile = "ensemble_" + TIME_DIR + '__' + wd + '__' + setting + ".cmd"
        cmd_list = []
        if os.path.isfile(outfile):
            print("WARNING! %s already exists" % outfile)
            continue
        for tid in openml_automl_benchmark:
            for seed in range(nseeds):
                es_dir_tmpl = [TIME_DIR, wd] + (['*'] * n_stars) + [setting + "_%d" % tid + "_%d" % seed + "*"]
                es_dir_tmpl = os.path.join(*es_dir_tmpl)
                es_dirs = glob.glob(es_dir_tmpl)
                for es_dir in es_dirs:
                    for es in ensemble_size:
                        for prt in performance_range_threshold:
                            for mkb in max_keep_best:
                                tmp = CMD
                                tmp += " --task-id %d" % tid
                                tmp += " --max-keep-best %s" % str(mkb)
                                tmp += " --ensemble-size %s" % str(es)
                                tmp += " --performance-range-threshold %s" % str(prt)
                                tmp += " --seed %d" % seed
                                tmp += " --ensemble-dir %s" % es_dir
                                cmd_list.append(tmp)
                                if only_portfolio:
                                    tmp += " --only-portfolio-runs"
                                    cmd_list.append(tmp)
        if len(cmd_list) > 0:
            with open(outfile, "w") as fh:
                for c in cmd_list:
                    fh.write(c)
                    fh.write("\n")
