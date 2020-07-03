import sys

sys.path.append('..')
from utils import RF, RFSH

CMD = "python ./run_autosklearn/create_run_autosklearn_with_portfolio.py " \
       "--nseeds %d --memory-limit %d --time-limit %d --per-run-time-limit %d --portfolio-dir %s --taskset openml_automl_benchmark"

# New baselines with preprocessing and other
NSEEDS = 10
PRTL = 60
TL = 600
MEMORY = 4000
WD = "10MIN_portfolio_runs"
PDIR = "./10MIN_portfolio/"
cmd = CMD % (NSEEDS, MEMORY, TL, PRTL, PDIR)

created = []
for set in (RF, RFSH):
    for s in set:
        if s in created:
            raise ValueError("Potential duplication %s" % s)
        created.append(s)
        print("%s --working-directory %s/%ssec/ --method %s  >> wportfolio_%s.cmd" % (cmd, WD, PRTL, s, s))
        print("%s --working-directory %s/NOPRT/ --method %s --use-prt-from-portfolio >> wportfolio_noprt_%s.cmd" % (cmd, WD, s, s))
sys.exit(1)
