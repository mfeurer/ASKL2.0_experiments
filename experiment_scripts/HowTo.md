# HowTo Run experiments

*Note:* Choose your setting. All following calls will **rely** on this variable being set correctly 
```
EXP=10MIN
EXP=20MIN
EXP=60MIN
EXP=10H
```

*Note:* each call to `master_calls.py` will create a new commands file. Execute all commands in that
file before moving to the next stage.

## Improvement 0
```
EXP=60MIN
python master_calls.py --do ASKL_improvement_zero --setting $EXP
```

## Improvement 1

ASKL on automldata

```
python master_calls.py --do ASKL_automldata --setting $EXP
```

## Improvement 2

1. ASKL on metadata
```
python master_calls.py --do ASKL_metadata --setting $EXP
```

2. ASKL on automldata

This is the same as what needs to be done for Improvement 1

```
python master_calls.py --do ASKL_automldata --setting $EXP
```

3. Create Matrix
```
python master_calls.py --do ASKL_getportfolio --setting $EXP
bash $EXP/ASKL_getportfolio/get_portfolio_configurations.cmd
```
First start one of the server commands, then the respective worker

*Note:* Usually, there are some jobs missing, so stop servers, restart them and submit again some workers

```
python master_calls.py --do run_create_matrix --setting $EXP
bash $EXP/ASKL_getportfolio/build_matrix.cmd
```

4. Create Portfolio
```
python master_calls.py --do run_create_symlinks --setting $EXP
python master_calls.py --do ASKL_create_portfolio --setting $EXP

First, submit these commands in ./$EXP/ASKL_create_portfolio/build_portfolio.cmd and ./$EXP/ASKL_create_portfolio/build_portfolio_cv.cmd. Afterwards, you can execute the commands in ./$EXP/ASKL_create_portfolio/execute_portfolio_cv.
```

5. ASKL+PO on automldata

```
python master_calls.py --do ASKL_run_with_portfolio  --setting $EXP
```

6. ASKL+KND on automldata (baseline)

```
python master_calls.py --do ASKL_automldata_w_ensemble_w_knd --setting $EXP

```

## Improvement 3

1. Train A^2 Selector

```
python master_calls.py --do AutoAuto_build  --setting $EXP
```

2. A^2SKL+PO on automldata

```
python master_calls.py --do AutoAuto_simulate  --setting $EXP
```

## Results

1. ASKL on automldata w/ ensembles
```
python master_calls.py --do ASKL_automldata_w_ensemble --setting $EXP
```

2. ASKL+PO on automldata w/ ensembles - RQ 2.1

```
python master_calls.py --do ASKL_run_with_portfolio_w_ensemble  --setting $EXP
```

3. ASKL for 10 seeds on all metadata sets (to build selector w/o portfolios) - RQ 2.2
```
python master_calls.py --do ASKL_metadata_full --setting $EXP
```

4. ASKL baselines for final comparison - includes ensemble data
```
python master_calls.py --setting $EXP --do ASKL_automldata_baseline_iter
python master_calls.py --setting $EXP --do ASKL_automldata_baseline_iter_no_metalearning
python master_calls.py --setting $EXP --do ASKL_automldata_baseline_iter_random
python master_calls.py --setting $EXP --do ASKL_automldata_baseline_full
python master_calls.py --setting $EXP --do ASKL_automldata_baseline_full_no_metalearning
python master_calls.py --setting $EXP --do ASKL_automldata_baseline_full_random
```

11. ASKL2 for final comparison

```
python master_calls.py --do RQ1_AutoAuto_build  --setting $EXP
python master_calls.py --do ASKL_run_with_portfolio_w_ensemble  --setting $EXP
python master_calls.py --do RQ1_AutoAuto_simulate  --setting $EXP
python master_calls.py --do RQ1_prune_run_with_portfolio  --setting $EXP
python master_calls.py --setting $EXP --do RQ1_AutoAuto_simulate_create_posthoc_symlinks
```

12. Build ensembles

```
python ensembles/ensemble_calls.py --time-dir $EXP
```

### RQ2.1 - are different evaluation strategies necessary for the selector?

1. Build the new selector
```
python master_calls.py --do RQ2.1_AutoAuto_build --setting ${EXP}
```

2. Build the symlinks
```
python master_calls.py --do RQ2.1_AutoAuto_simulate --setting ${EXP}
```

3. Create a symlink to avoid a bug related to the directory structure
```
cd 60MIN/ASKL_automldata_w_ensemble
ln -s . 360
```

### RQ2.2 - do we need the portfolios

1. Build selectors
```
python master_calls.py --do RQ2.2_AutoAuto_build --setting ${EXP}
```

2. Simulate AutoAutoML
```
python master_calls.py --do RQ2.2_AutoAuto_simulate --setting ${EXP}
```

### RQ2.3 - is the selector necessary at all?

```
python master_calls.py --do RQ2.3_AutoAuto_build --setting $EXP
```

```
python master_calls.py --do RQ2.3_AutoAuto_simulate --setting ${EXP}
```
