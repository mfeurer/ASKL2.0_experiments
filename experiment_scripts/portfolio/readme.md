# What to run and execute

1. Create the commands file
```bash
for holdout_strategy in 3CV 5CV 10CV holdout
do
python /home/feurerm/projects/2020_IEEE_Autosklearn_experiments/experiment_scripts/portfolio/create_build_portfolio.py \
--input-directory /home/eggenspk/PoSHAutosklearn/2020_IEEE_Autosklearn_experiments/experiment_scripts/10MIN_metadata/RF_None_${holdout_strategy}_iterative_es_if/ \
--n-seeds 10 --portfolio-size 32 --max-runtime 600 --max-execution-runtime 600 \
--output-directory /home/eggenspk/PoSHAutosklearn/2020_IEEE_Autosklearn_experiments/experiment_scripts/10MIN_portfolio/${holdout_strategy}

python /home/feurerm/projects/2020_IEEE_Autosklearn_experiments/experiment_scripts/portfolio/create_build_portfolio.py \
--input-directory /home/eggenspk/PoSHAutosklearn/2020_IEEE_Autosklearn_experiments/experiment_scripts/10MIN_metadata/RF_None_${holdout_strategy}_iterative_es_if/ \
--n-seeds 10 --portfolio-size 32 --max-execution-runtime 600 \
--output-directory /home/eggenspk/PoSHAutosklearn/2020_IEEE_Autosklearn_experiments/experiment_scripts/10MIN_portfolio/${holdout_strategy}
done
```

2. Submit the following to build the portfolio:
```bash
cd portfolio
bash compile.sh
for holdout_strategy in 3CV 5CV 10CV holdout
do
python /home/eggenspk/scripts/metahelper/slurm_helper.py -q bosch_cpu-cascadelake \
--timelimit 15000 --memory_per_job 6000 \
--startup /home/eggenspk/PoSHAutosklearn/2020_IEEE_Autosklearn_experiments/experiment_scripts/startup.sh \
/home/eggenspk/PoSHAutosklearn/2020_IEEE_Autosklearn_experiments/experiment_scripts/10MIN_portfolio/${holdout_strategy}/build_portfolio_32_None.cmd \
-o ./10MIN_portfolio_sgeout/ -l ./10MIN_portfolio_sgeout/

python /home/eggenspk/scripts/metahelper/slurm_helper.py -q bosch_cpu-cascadelake \
--timelimit 15000 --memory_per_job 6000 \
--startup /home/eggenspk/PoSHAutosklearn/2020_IEEE_Autosklearn_experiments/experiment_scripts/startup.sh \
/home/eggenspk/PoSHAutosklearn/2020_IEEE_Autosklearn_experiments/experiment_scripts/10MIN_portfolio/${holdout_strategy}/build_portfolio_32_600.cmd \
-o ./10MIN_portfolio_sgeout/ -l ./10MIN_portfolio_sgeout/

python /home/eggenspk/scripts/metahelper/slurm_helper.py -q bosch_cpu-cascadelake \
--timelimit 15000 --memory_per_job 6000 \
--startup /home/eggenspk/PoSHAutosklearn/2020_IEEE_Autosklearn_experiments/experiment_scripts/startup.sh \
/home/eggenspk/PoSHAutosklearn/2020_IEEE_Autosklearn_experiments/experiment_scripts/10MIN_portfolio/${holdout_strategy}/build_portfolio_cv_32_None.cmd \
-o ./10MIN_portfolio_sgeout/ -l ./10MIN_portfolio_sgeout/

python /home/eggenspk/scripts/metahelper/slurm_helper.py -q bosch_cpu-cascadelake \
--timelimit 15000 --memory_per_job 6000 \
--startup /home/eggenspk/PoSHAutosklearn/2020_IEEE_Autosklearn_experiments/experiment_scripts/startup.sh \
/home/eggenspk/PoSHAutosklearn/2020_IEEE_Autosklearn_experiments/experiment_scripts/10MIN_portfolio/${holdout_strategy}/build_portfolio_cv_32_600.cmd \
-o ./10MIN_portfolio_sgeout/ -l ./10MIN_portfolio_sgeout/
done
```

3. Afterwards, submit the following to virtually execute the portfolios
```bash
cd portfolio
bash compile.sh
for holdout_strategy in 3CV 5CV 10CV holdout
do

python /home/eggenspk/scripts/metahelper/slurm_helper.py -q bosch_cpu-cascadelake \
--timelimit 15000 --memory_per_job 6000 \
--startup /home/eggenspk/PoSHAutosklearn/2020_IEEE_Autosklearn_experiments/experiment_scripts/startup.sh \
/home/eggenspk/PoSHAutosklearn/2020_IEEE_Autosklearn_experiments/experiment_scripts/10MIN_portfolio/${holdout_strategy}/execute_portfolio_cv_32_None.cmd \
-o ./10MIN_portfolio_sgeout/ -l ./10MIN_portfolio_sgeout/

python /home/eggenspk/scripts/metahelper/slurm_helper.py -q bosch_cpu-cascadelake \
--timelimit 15000 --memory_per_job 6000 \
--startup /home/eggenspk/PoSHAutosklearn/2020_IEEE_Autosklearn_experiments/experiment_scripts/startup.sh \
/home/eggenspk/PoSHAutosklearn/2020_IEEE_Autosklearn_experiments/experiment_scripts/10MIN_portfolio/${holdout_strategy}/execute_portfolio_cv_32_600.cmd \
-o ./10MIN_portfolio_sgeout/ -l ./10MIN_portfolio_sgeout/
done
```
