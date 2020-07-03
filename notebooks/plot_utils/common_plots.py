import matplotlib.pyplot as plt
import itertools

import collections
import numpy as np
import pandas as pd

from .style import style_dc


def average_plot(model_list, res_dc, valid_pretty, horizon, task_ids_sorted_by_num_features,
                 min_diff_dc, figsize=None, legend=True):
    if not figsize:
        figsize = [8, 6]
    plt.figure(figsize=figsize)
    bounds = [
        [0, len(task_ids_sorted_by_num_features)],
        [-5, len(task_ids_sorted_by_num_features)],
        [0, 5],
    ]

    for low, up in bounds:
        all_results = collections.defaultdict(list)
        for tid in task_ids_sorted_by_num_features[low:up]:
            min_for_task = min_diff_dc[tid][0]
            diff = min_diff_dc[tid][1]
            for mode in model_list:
                tmp = pd.DataFrame(res_dc[horizon][tid][mode]).sort_index(axis=1).ffill(axis=1).mean()
                tmp[horizon * 60.] = np.NaN
                tmp = tmp.ffill()
                tmp = (tmp - min_for_task) / diff
                all_results[mode].append(tmp)

        colors = itertools.cycle(style_dc["colors"])

        for mode in model_list:
            c = next(colors)
            tmp = pd.DataFrame(all_results[mode]).ffill(axis=1)
            tmp = tmp.mean()
            x = tmp.index
            y = tmp.to_numpy()
            if low == 0 and up == len(task_ids_sorted_by_num_features):
                label = valid_pretty[horizon][mode]
                linestyle = "-"
                alpha = 1
            elif low == -5 and up == len(task_ids_sorted_by_num_features):
                label = None
                linestyle = "--"
                alpha = 0.5
            elif low == 0 and up == 5:
                label = None
                linestyle = ":"
                alpha = 0.5
            plt.plot(x / 60, y, label=label, linewidth=style_dc["linewidth"], linestyle=linestyle,
                     alpha=alpha, c=c)

    if legend:
        plt.legend(fontsize=style_dc["fontsize"])
    plt.xticks(fontsize=style_dc["fontsize"])
    plt.yticks(fontsize=style_dc["fontsize"])
    plt.xlabel('time [min]', fontsize=style_dc["fontsize"])
    plt.ylabel("normalized BER", fontsize=style_dc["fontsize"])
    plt.xlim(0, horizon)


def rank(model_list, res_dc, valid_pretty, horizon, task_ids_sorted_by_num_features,
         n_iter=200, steplength=5, paired=False, legend=True, figsize=None):
    # Ranking Plot (adapted from https://github.com/automl/auto-sklearn/blob/master/scripts/2015_nips_paper/plot/plot_ranks.py#L121)

    # Step 1. Merge all trajectories into one Dataframe object.
    #####################################################################################
    all_trajectories = []
    print("Plotting %d models" % len(model_list))

    steps = []
    for minute in range(horizon):
        for second in range(0, 60, steplength):
            timestep = minute * 60 + second
            steps.append(timestep)
    steps.append(horizon * 60)

    for mode in model_list:
        trajectories = []
        for tid in task_ids_sorted_by_num_features:
            # Get data from csv
            for i in range(len(res_dc[horizon][tid][mode])):
                res_dc[horizon][tid][mode][i][0] = 1.0
                for step in steps:
                    if step > 0:
                        res_dc[horizon][tid][mode][i][step] = np.NaN
            a = pd.DataFrame(res_dc[horizon][tid][mode]).sort_index(axis=1).ffill(axis=1)
            a = a.transpose()
            a = a.loc[steps]
            trajectories.append(a)

        all_trajectories.append(trajectories)

    # Step 2. Compute average ranks of the trajectories.
    #####################################################################################
    all_rankings = []
    n_tasks = len(task_ids_sorted_by_num_features)

    if paired:
        n_iter = all_trajectories[0][0].shape[1]

    for i in range(n_iter):
        if i % 50 == 0: print("%d / %d" % (i, n_iter))
        if paired:
            pick = np.ones(len(model_list), dtype=np.int) * i
        else:
            pick = np.random.choice(all_trajectories[0][0].shape[1],
                                    size=(len(model_list)))
        for j in range(n_tasks):
            all_trajectories_tmp = pd.DataFrame(
                {model_list[k]: at[j].iloc[:, pick[k]] for
                 k, at in enumerate(all_trajectories)}
            )
            all_trajectories_tmp = all_trajectories_tmp.fillna(method='ffill', axis=0)
            r_tmp = all_trajectories_tmp.rank(axis=1)
            all_rankings.append(r_tmp)

    final_ranks = []
    for i, model in enumerate(model_list):
        ranks_for_model = []
        for ranking in all_rankings:
            ranks_for_model.append(ranking.loc[:, model])
        ranks_for_model = pd.DataFrame(ranks_for_model)
        ranks_for_model = ranks_for_model.fillna(method='ffill', axis=1)
        final_ranks.append(ranks_for_model.mean(skipna=True))

    # Step 3. Plot the average ranks over time.
    ######################################################################################
    if not figsize:
        figsize = (10, 5)
    plt.figure(figsize=figsize)
    colors = itertools.cycle(style_dc["colors"])
    for i, model in enumerate(model_list):
        X_data = []
        y_data = []
        for x, y in final_ranks[i].iteritems():
            X_data.append(x)
            y_data.append(y)
        X_data.append(horizon * 60)
        X_data = np.array(X_data) / 60
        y_data.append(y)
        plt.plot(X_data, y_data, label=valid_pretty[horizon][model], c=next(colors),
                 linewidth=style_dc["linewidth"])
    plt.xlabel('time [min]', fontsize=style_dc["fontsize"])
    plt.ylabel('average rank', fontsize=style_dc["fontsize"])
    if legend:
        plt.legend(fontsize=style_dc["fontsize"])
    plt.xticks(fontsize=style_dc["fontsize"])
    plt.yticks(fontsize=style_dc["fontsize"])
    plt.xlim(0, horizon)
